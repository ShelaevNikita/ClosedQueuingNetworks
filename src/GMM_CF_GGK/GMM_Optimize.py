
import numpy as np
import GMM_CF_GGK

from re             import sub
from scipy.linalg   import solve
from scipy.optimize import NonlinearConstraint, minimize

FileNameGMM      = './src/GMM_CF_GGK/InputData.dat'
FileNameOptimize = './src/GMM_CF_GGK/InputData_Optimize.dat'

class GMM_Optimize():

    S = 1.       # Лимит ограничений
    T = 1        # Максимальное количество итераций оптимизации
    M = 1        # Количество узлов (приборов) в сети
    R = 1        # Количество классов заявок

    ParameterDict = { 'C' : np.empty(1),  # Ценность заявок определённого класса (размер - R)
                      'A' : np.empty(1),  # Коэффициент для нелинейных ограничений (размер - M)
                      'K' : np.empty(1),  # Число каналов обслуживания в узлах сети (размер - M)
                      'D' : [] }          # Матрица для определения начала и конца цикла обслуживания заявок (размер - R x M)

    def __init__(self, inputFileNameGMM, inputFileNameOptimize):
        self.classGMM = GMM_CF_GGK.GMM_CF_GGK(inputFileNameGMM, False)
        self.main(inputFileNameGMM, inputFileNameOptimize)

    # Получение из входного файла значений параметров ограничений
    def getInputParameter(self, inputFile):
        flagD  = False
        for line in inputFile:
            lineParamIndex = line.find('=')
            if lineParamIndex > -1 or flagD == True:
                if flagD:
                    lineParamIndex = 0
                lineParam  = sub('[^0-9\.]', ' ', line[lineParamIndex:])
                paramArray = sub(r'\s+', ' ', lineParam.strip()).split()
                paramName  = line[(lineParamIndex - 3):(lineParamIndex - 1)].strip()
                if paramName == 'D':
                    flagD = True
                try:
                    if   paramName == 'S':
                        self.S = float(paramArray[0])
                    elif paramName == 'T':
                        self.T = int(paramArray[0])
                    elif paramName in ['C', 'A']:
                        self.ParameterDict[paramName] = np.array(list(map(float, paramArray)))
                    elif flagD == True:
                        self.ParameterDict['D'].append(list(map(float, paramArray)))
                        if len(self.ParameterDict['D']) == self.R:
                            flagD = False
                except TypeError:
                    print(f'\n   Error! TypeError with parameter "{paramName}"...')
                    continue
        return

    # Открытие файла с заданными параметрами ограничений
    def splitInputFile(self, inputFileName):
        try:
            inputFile = open(inputFileName, 'r', encoding = 'utf-8')
            self.getInputParameter(inputFile)
            inputFile.close()
        except FileNotFoundError:
            print(f'\n   ERROR! Requested file "{inputFileName}" not found!\n')
            return
        return

    # Получение значений параметров сети
    def getParameterGMM(self, inputFileNameGMM):
        self.classGMM.splitInputFile(inputFileNameGMM)
        self.classGMM.findFactorial()
        res = self.classGMM.returnParameterDict()
        self.M = res['M']
        self.R = res['R']
        self.ParameterDict['K'] = res['K']
        return np.array(res['MU']).reshape(self.M * self.R)

    # Вычисление ценности
    def funCost(self, MU):
        self.classGMM.ParameterDict['MU'] = MU.reshape(self.R, self.M)
        lambdaArr, _, _, _, _, _, _ = self.classGMM.forIter(self.classGMM.findMostLoadedNode())
        cost = 0.
        for i in range(self.M):
            cost += sum([self.ParameterDict['C'][r] * self.ParameterDict['D'][r][i] * lambdaArr[i][r]
                         for r in range(self.R)])
        print(cost)
        return -cost

    # Функция ограничений
    def constraint(self, MU):
        self.classGMM.ParameterDict['MU'] = MU.reshape(self.R, self.M)
        _, _, _, _, MUa, _, _ = self.classGMM.forIter(self.classGMM.findMostLoadedNode())
        return sum([self.ParameterDict['K'][i] * (MUa[i] ** self.ParameterDict['A'][i]) for i in range(self.M)])

    # Оптимизация MU
    def optimize(self, MU):
        bounds = [(0., None) for i in range(self.M * self.R)]
        constraints = [NonlinearConstraint(self.constraint, self.S, self.S)]
        MU_opt = minimize(self.funCost, MU, method = 'SLSQP', bounds = bounds, 
                          constraints = constraints, options = {'maxiter': self.T, 'ftol': 1e-5})
        return MU_opt.x

    # Главная функция
    def main(self, inputFileNameGMM, inputFileNameOptimize):
        startMU = self.getParameterGMM(inputFileNameGMM)
        self.splitInputFile(inputFileNameOptimize)
        optMU = self.optimize(startMU).reshape(self.R, self.M)
        print('\n   MU (optimal value) =', optMU, '\n')
        self.classGMM.ParameterDict['MU'] = optMU
        lambdaArray, jArray, t, to, MU, Q, CSi = self.classGMM.forIter(self.classGMM.findMostLoadedNode())
        t = np.transpose(np.array(t))
        self.classGMM.find_All(np.transpose(lambdaArray), jArray, t, to, MU, Q, CSi)
        return 0

if __name__ == '__main__':
    GMM_Optimize(FileNameGMM, FileNameOptimize)
