
class utilidad:
    def __init__(self):
        print()

    def acuracia(self, fp, fn, vp):
            return (vp + self) / (vp + self + fp + fn)

    def especificidade(vn, fp):
            return (vn) / (vn + fp)

    def sensibilidade(fn, vp):
            return ((vp) / (vp + fn))

    def main_validacao(self, fp, fn, vp):
        """
        :param vn: verdadeiros positivos
        :param fp: falsos positivos
        :param fn: falsos negativos
        :param vp: verdadeiros positivos
        :return:
        """
        print('vp = ', vp, ' fp = ', fp, ' fn = ', fn, ' vp = ', vp)
        print(' agudeza______________[ %.2f%s ]' % ((self.acuracia(self, fp, fn, vp) * 100), "%"))
        print(' Especifidad________[ %.2f%s ]' % ((self.especificidade(self, fp) * 100), "%"))
        print(' Sensibilidad_________[ %.2f%s ]' % ((self.sensibilidade(fn, vp) * 100), "%"))