"""工具验收用的对照。保存路由器的两个输出，在副本里故意改动一个专家编号。"""


def setup(model):
    def capture(module, args, result):
        global ids, weights
        ids, weights = [item.detach().clone() for item in result]

    model.get_submodule("model.layers.1.mlp.gate").register_forward_hook(capture)


def probe(model, step, inputs, output):
    changed = ids.clone()
    changed[0, 0] += 1
    return {"ids": ids, "weights": weights, "changed": changed}
