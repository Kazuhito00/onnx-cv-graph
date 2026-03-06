"""複数 OnnxGraphOp を直列合成して1つの ONNX モデルにする ChainOp."""

from typing import Dict, List

from onnx import GraphProto, TensorProto, helper

from src.base import OnnxGraphOp, ParamMeta, TensorSpec


class ChainOp(OnnxGraphOp):
    """複数の OnnxGraphOp を直列合成する.

    各 op の build_graph() から得たグラフをプレフィックス付きで結合し、
    中間出力→次段入力を内部接続する.

    自動エクスポート対象外: variants() は空リストを返す.
    """

    def __init__(self, ops: List[OnnxGraphOp]):
        """合成対象の op リストを受け取る.

        Parameters
        ----------
        ops : list[OnnxGraphOp]
            直列合成する op のリスト (先頭→末尾の順で処理される)

        Raises
        ------
        ValueError
            op が2つ未満の場合
        """
        if len(ops) < 2:
            raise ValueError("ChainOp には2つ以上の op が必要です")
        self._ops = ops

    @property
    def op_name(self) -> str:
        return "_".join(op.op_name for op in self._ops)

    @property
    def input_specs(self) -> List[TensorSpec]:
        """先頭 op の画像入力 + 全 op のパラメータ入力を返す."""
        specs: List[TensorSpec] = []
        # 先頭 op の画像入力
        for spec in self._ops[0].input_specs:
            if spec[0] == "input":
                specs.append(spec)
                break
        # 全 op のパラメータ入力 (画像入力以外)
        param_names = self._collect_param_names()
        for renamed, (op, orig_spec) in param_names.items():
            specs.append((renamed, orig_spec[1], orig_spec[2]))
        return specs

    @property
    def output_specs(self) -> List[TensorSpec]:
        return self._ops[-1].output_specs

    @property
    def param_meta(self) -> Dict[str, ParamMeta]:
        """全 op の param_meta をリネーム後のキーでマージ."""
        result: Dict[str, ParamMeta] = {}
        param_names = self._collect_param_names()
        for renamed, (op, orig_spec) in param_names.items():
            orig_name = orig_spec[0]
            if orig_name in op.param_meta:
                result[renamed] = op.param_meta[orig_name]
        return result

    @classmethod
    def variants(cls) -> "List[OnnxGraphOp]":
        """自動エクスポート対象外."""
        return []

    def _collect_param_names(self) -> Dict[str, "tuple[OnnxGraphOp, TensorSpec]"]:
        """全 op のパラメータ入力を収集し、衝突時はリネームする.

        Returns
        -------
        dict[str, tuple[OnnxGraphOp, TensorSpec]]
            リネーム後の名前 → (元の op, 元の TensorSpec)
        """
        image_inputs = {"input", "input2"}
        result: Dict[str, tuple[OnnxGraphOp, TensorSpec]] = {}
        seen_names: set[str] = set()

        for op in self._ops:
            for spec in op.input_specs:
                if spec[0] in image_inputs:
                    continue
                name = spec[0]
                if name in seen_names:
                    # 衝突時は op_name.param にリネーム
                    name = f"{op.op_name}.{spec[0]}"
                seen_names.add(name)
                result[name] = (op, spec)
        return result

    def build_graph(self) -> GraphProto:
        """各 op のグラフをプレフィックス付きで結合する."""
        all_nodes = []
        all_initializers = []
        image_inputs = {"input", "input2"}

        # パラメータ名マッピング: (op_index, 元の名前) → リネーム後の名前
        param_rename_map: Dict[tuple[int, str], str] = {}
        param_names = self._collect_param_names()
        for renamed, (op, orig_spec) in param_names.items():
            op_idx = self._ops.index(op)
            param_rename_map[(op_idx, orig_spec[0])] = renamed

        for i, op in enumerate(self._ops):
            graph = op.build_graph()
            prefix = f"{i}_"
            is_first = i == 0
            is_last = i == len(self._ops) - 1

            # この op のパラメータ入力名を収集
            op_param_names = {
                spec[0] for spec in op.input_specs if spec[0] not in image_inputs
            }

            # 名前のリマッピングテーブルを構築
            name_map: Dict[str, str] = {}
            for node in graph.node:
                for tensor_name in list(node.input) + list(node.output):
                    if tensor_name in name_map:
                        continue
                    if tensor_name == "input" and is_first:
                        # 先頭 op の画像入力はそのまま
                        name_map[tensor_name] = "input"
                    elif tensor_name == "input" and not is_first:
                        # 中間 op の画像入力 → 前段の出力に接続
                        name_map[tensor_name] = f"{i - 1}_output"
                    elif tensor_name == "output" and is_last:
                        # 末尾 op の出力はそのまま "output"
                        name_map[tensor_name] = "output"
                    elif tensor_name in op_param_names:
                        # パラメータ入力はリネーム後の名前を使う
                        name_map[tensor_name] = param_rename_map.get(
                            (i, tensor_name), tensor_name
                        )
                    else:
                        # その他のテンソルにはプレフィックスを付与
                        name_map[tensor_name] = prefix + tensor_name

            # initializer の名前もリマップ対象に追加
            for init in graph.initializer:
                if init.name not in name_map:
                    name_map[init.name] = prefix + init.name

            # ノードの入出力名をリマップ
            for node in graph.node:
                new_node = helper.make_node(
                    node.op_type,
                    inputs=[name_map.get(n, prefix + n) for n in node.input],
                    outputs=[name_map.get(n, prefix + n) for n in node.output],
                    name=prefix + (node.name or node.op_type),
                    **{attr.name: helper.get_attribute_value(attr) for attr in node.attribute},
                )
                all_nodes.append(new_node)

            # initializer をリマップ
            for init in graph.initializer:
                init.name = name_map.get(init.name, prefix + init.name)
                all_initializers.append(init)

        # グラフ入力を構築
        graph_inputs = []
        # 先頭 op の画像入力
        for spec in self._ops[0].input_specs:
            if spec[0] == "input":
                graph_inputs.append(
                    helper.make_tensor_value_info(spec[0], spec[1], spec[2])
                )
                break
        # パラメータ入力
        for renamed, (op, orig_spec) in param_names.items():
            graph_inputs.append(
                helper.make_tensor_value_info(renamed, orig_spec[1], orig_spec[2])
            )

        # グラフ出力
        last_out = self._ops[-1].output_specs[0]
        graph_outputs = [
            helper.make_tensor_value_info("output", last_out[1], last_out[2])
        ]

        return helper.make_graph(
            all_nodes,
            self.op_name,
            graph_inputs,
            graph_outputs,
            initializer=all_initializers,
        )
