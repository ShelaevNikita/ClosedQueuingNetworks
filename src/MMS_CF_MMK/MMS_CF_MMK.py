
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './src/MMS_CF_MMK/InputData.dat'

class MMS_CF_MMK():

    M = 1            # Количество узлов (приборов) в сети
    N = 1            # Количество заявок в сети
    ErrorRate = 0.1  # Погрешность выполнения программы

    ParameterDict = { 'Q' : [],            # Матрица передач, определяющая маршрутизацию заявок в сети (размер - M x M)
                      'K' : np.empty(1),   # Число каналов обслуживания в узлах сети (размер - M)
                      'MU': np.empty(1),   # Интенсивность обслуживания заявок в узлах сети (размер - M)                          
                      'W' : np.empty(1),   # Вероятность нахождения заявки в текущем узле сети (размер - M)
                      'F' : np.empty(1) }  # Массив значений факториалов (размер - M)
    
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
                    elif paramName == 'K':
                        self.ParameterDict['K']  = np.array(list(map(int, paramArray)))
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
            if self.M == 0 or self.N == 0 or min(self.ParameterDict['K']) == 0 or min(self.ParameterDict['MU']) == 0:
                raise ValueError
            self.ParameterDict['K']  = np.array(self.ParameterDict['K']).reshape(self.M)
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
            value = self.ParameterDict['W'][i] / (self.ParameterDict['MU'][i] * self.ParameterDict['K'][i])
            if value > max:
                max = value
                indexMax = i
        return indexMax

    # Поиск Ji
    def findJi(self, lambdai, i):
        mui = self.ParameterDict['MU'][i]
        k = self.ParameterDict['K'][i]
        p = lambdai / mui
        sumH = sum([(p ** j) * 1 / self.ParameterDict['F'][j] for j in range(1, (k + 1))])
        P0 = 1 / (1 + sumH + (p ** (k + 1)) / (self.ParameterDict['F'][k] * (k - p)))
        toi = (p ** k) * P0 / (k * self.ParameterDict['F'][k] * mui * ((1 - p / k) ** 2))
        hh = self.N / (self.N + toi * mui) * (self.N - 1) / self.N * toi
        ni = lambdai * (hh + 1 / mui)
        return (ni - p, ni)

    # Выполнение одной итерации
    def doOneIter(self, Bi):
        lambdaArray = [Bi * elem for elem in self.ParameterDict['W']]
        jArray = []
        noArray = []
        for i in range(self.M):
            no, ni = self.findJi(lambdaArray[i], i)
            jArray.append(ni)
            noArray.append(no)
        return (lambdaArray, jArray, noArray)

    # Функция метода Вегстейна
    def funWegstein(self, iter):
        yk = self.yArray[iter:(iter + 2)]
        xk = self.xArray[(iter - 1):(iter + 1)]
        BiNew = yk[1] - ((yk[1] - yk[0]) * (yk[1] - xk[1])) / (yk[1] - yk[0] - xk[1] + xk[0])
        return BiNew

    # Выполнение всех итераций
    def forIter(self, indexMax):
        iter = -1
        self.ParameterDict['F'] = [np.math.factorial(j) for j in range(max(self.ParameterDict['K']) + 1)]
        Bi = (self.ParameterDict['K'][indexMax] * (1 - 1 / self.N) * \
            self.ParameterDict['MU'][indexMax]) / self.ParameterDict['W'][indexMax]
        self.xArray.append(Bi)
        self.yArray.append(Bi)
        L = 0.
        while abs(L - self.N) > self.ErrorRate:
            iter = iter + 1
            lambdaArray, jArray, no = self.doOneIter(Bi)
            L = sum(jArray)
            Bi = Bi * self.N / L
            self.yArray.append(Bi)
            if iter > 0:
                Bi = self.funWegstein(iter)
            self.xArray.append(Bi)
        return (lambdaArray, jArray, no)

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
    def find_All(self, lambdaArray, jArray, no):
        t = [jArray[i] / lambdaArray[i] for i in range(self.M)]
        to = [t[i] - 1 / self.ParameterDict['MU'][i] for i in range(self.M)]
        V = [self.N / lambdaArray[i] for i in range(self.M)]
        fi = self.M / sum(V)
        self.printResult(lambdaArray, jArray, t, V, no, to, fi)
        return

    # Главная функция
    def main(self, inputFileName):
        if self.splitInputFile(inputFileName) == -1:
            return 1
        lambdaArray, jArray, no = self.forIter(self.findMostLoadedNode())
        self.find_All(lambdaArray, jArray, no)
        return 0

if __name__ == '__main__':
    MMS_CF_MMK(FileName)