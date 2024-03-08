class InnerFnOpsHandler:

    def __init__(self, imp):
        self.imp = imp

    def load(self, name: str, index):
        return self.imp.ir_builder.create_load_f32_op(self.imp.unknown_loc,
                                                      name, index).result(0)

    def sin(self, x0):
        return self.imp.ir_builder.create_sin_op(self.imp.unknown_loc,
                                                 x0).result(0)

    def cos(self, x0):
        return self.imp.ir_builder.create_cos_op(self.imp.unknown_loc,
                                                 x0).result(0)

    def add(self, x0, x1):
        # FIXME: distinguish int add or float add
        return self.imp.ir_builder.create_addf_op(self.imp.unknown_loc, x0,
                                                  x1).result(0)
