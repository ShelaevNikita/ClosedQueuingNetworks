
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './src/MSS_CF_MM1/InputData.dat'

class MSS_CF_MM1():

    M = 1            # Количество узлов (приборов) в сети
    N = 1            # Количество заявок в сети
    ErrorRate = 0.1  # Погрешность выполнения программы

    ParameterDict = { 'Q' : [],              # Матрица передач, определяющая маршрутизацию заявок в сети (размер - M x M)
                      'MU': np.empty(1),     # Интенсивность обслуживания заявок в узлах сети (размер - M)                          
                      'W' : np.empty(1) }    # Вероятность нахождения заявки в текущем узле сети (размер - M)

    # 2 Массива для метода Вегстейна
    xArray = []
    yArray = []

    def __init__(self, inputFileName):
        self.main(inputFileName)

    # Получение из входного файла значений параметров сети
    def getInputParameter(self, inputFile):
        flagQ = False
        for line in inputFile:
            lineParamIndex = line.find('=')
            if lineParamIndex > -1 or flagQ == True:
                if flagQ:
                    lineParamIndex = 0
                lineParam  = sub('[^0-9\.]', ' ', line[lineParamIndex:])
                paramArray = sub(r'\s+', ' ', lineParam.strip()).split()
                paramName  = line[(lineParamIndex - 3):(lineParamIndex - 1)].strip()
                if paramName == 'Q':
                    flagQ = True
                try:
                    if   paramName == 'M':
                        self.M = int(paramArray[0])
                    elif paramName == 'N':
                        self.N = int(paramArray[0])
                    elif paramName == 'E':
                        self.ErrorRate = float(paramArray[0])
                    elif paramName == 'MU':
                        self.ParameterDict['MU'] = np.array(list(map(float, paramArray)))
                    elif flagQ == True:
                        self.ParameterDict['Q'].append(list(map(float, paramArray)))
                        if len(self.ParameterDict['Q']) == self.M:
                            flagQ = False
                except TypeError:
                    print(f'\n   ERROR! TypeError with parameter "{paramName}"...\n')
                    return -1
        try:
            if self.M == 0 or self.N == 0 or min(self.ParameterDict['MU']) == 0:
                raise ValueError
            self.ParameterDict['Q']  = np.array(self.ParameterDict['Q']).reshape(self.M, self.M)
            self.ParameterDict['MU'] = np.array(self.ParameterDict['MU']).reshape(self.M)
        except ValueError:
            print('\n   ERROR! Incorrect input data format\n')
            return -1
        return

    # Открытие файла с заданными параметрами сети
    def splitInputFile(self, inputFileName):
        res = -1
        try:
            inputFile = open(inputFileName, 'r', encoding = 'utf-8')
            res = self.getInputParameter(inputFile)
            inputFile.close()
        except FileNotFoundError:
            print(f'\n   ERROR! Requested file "{inputFileName}" not found!\n')
            return res
        return res

    # Поиск массива W
    def findWArray(self):
        flag = False
        for elem in self.ParameterDict['Q']:
            if not 1. in elem:
                flag = True
                break     
        if flag:
            A = np.copy(np.transpose(self.ParameterDict['Q']))
            B = np.zeros(self.M)
            for i in range(self.M):
                A[i][i] = A[i][i] - 1
            A[-1] = np.ones(self.M)
            B[-1] = 1
            self.ParameterDict['W'] = solve(A, B)
        else:
            self.ParameterDict['W'] = np.ones(self.M)
        return

    # Поиск наиболее загруженного узла в сети
    def findMostLoadedNode(self):
        self.findWArray()
        max = 0.
        indexMax = 0
        for i in range(self.M):
            value = self.ParameterDict['W'][i] / self.ParameterDict['MU'][i]
            if value > max:
                max = value
                indexMax = i
        return indexMax

    # Выполнение одной итерации
    def doOneIter(self, Bi):
        lambdaArray = [Bi * elem for elem in self.ParameterDict['W']]
        jArray = []
        for i in range(self.M):
            lambdai = lambdaArray[i]
            mui = self.ParameterDict['MU'][i]
            kk = lambdai / (mui - lambdai)
            hh = self.N / (self.N + kk) * kk / mui * (self.N - 1) / self.N
            jArray.append(lambdai * (hh + 1 / mui))
        return (lambdaArray, jArray)

    # Функция метода Вегстейна
    def funWegstein(self, iter):
        yk = self.yArray[iter:(iter + 2)]
        xk = self.xArray[(iter - 1):(iter + 1)]
        BiNew = yk[1] - ((yk[1] - yk[0]) * (yk[1] - xk[1])) / (yk[1] - yk[0] - xk[1] + xk[0])
        return BiNew

    # Выполнение всех итераций
    def forIter(self, indexMax):
        iter = -1
        Bi = self.ParameterDict['MU'][indexMax] / self.ParameterDict['W'][indexMax] * (1 - 1 / self.N)
        self.xArray.append(Bi)
        self.yArray.append(Bi)
        L = 0.
        while abs(L - self.N) > self.ErrorRate:
            iter = iter + 1
            lambdaArray, jArray = self.doOneIter(Bi)
            L = sum(jArray)
            Bi = Bi * self.N / L
            self.yArray.append(Bi)
            if iter > 0:
                Bi = self.funWegstein(iter)
            self.xArray.append(Bi)
        return (lambdaArray, jArray)

    # Отображение полученного результата
    def printResult(self, lambdaArray, jArray, t, V, no, to, fi):
        print('\n   lambda =', lambdaArray)
        print('\n   n =', jArray)
        print('\n   t =', t)
        print('\n   V =', V)
        print('\n   no =', no)
        print('\n   to =', to)
        print('\n   fi =', fi, '\n')
        return

    # Вычисление других характеристик сети
    def find_All(self, lambdaArray, jArray):
        t = [jArray[i] / lambdaArray[i] for i in range(self.M)]
        to = [t[i] - 1 / self.ParameterDict['MU'][i] for i in range(self.M)]
        no = [jArray[i] * to[i] / t[i] for i in range(self.M)]
        V = [self.N / lambdaArray[i] for i in range(self.M)]
        fi = self.M / sum(V)
        self.printResult(lambdaArray, jArray, t, V, no, to, fi)
        return

    # Главная функция
    def main(self, inputFileName):
        if self.splitInputFile(inputFileName) == -1:
            return 1
        lambdaArray, jArray = self.forIter(self.findMostLoadedNode())
        self.find_All(lambdaArray, jArray)
        return 0

if __name__ == '__main__':
    MSS_CF_MM1(FileName)