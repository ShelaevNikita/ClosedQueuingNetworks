
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './src/GMS_CF_GGK/InputData.dat'

class GMS_CF_GGK():

    M = 1            # Количество узлов (приборов) в сети
    N = 1            # Количество заявок в сети
    ErrorRate = 0.1  # Погрешность выполнения программы

    ParameterDict = { 'Q' : [],   # Матрица передач, определяющая маршрутизацию заявок в сети (размер - M x M)
                      'K' : [],   # Число каналов обслуживания в узлах сети (размер - M)
                      'MU': [],   # Интенсивность обслуживания заявок в узлах сети (размер - M)                          
                      'CS': [],   # Квадраты коэффициентов вариации длительностей обслуживания заявок в узлах сети (размер - M)
                      'W' : [],   # Вероятность нахождения заявки в текущем узле сети (размер - M)
                      'F' : [] }  # Массив значений факториалов (размер - M)

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
                        self.ErrorRate = abs(float(paramArray[0]))
                    elif paramName == 'K':
                        self.ParameterDict['K']  = np.array(list(map(int, paramArray)))
                    elif paramName in ['CS', 'MU']:
                        self.ParameterDict[paramName] = np.array(list(map(float, paramArray)))                 
                    elif flagQ == True:
                        self.ParameterDict['Q'].append(np.array(list(map(float, paramArray))))
                        if len(self.ParameterDict['Q']) == self.M:
                            flagQ = False
                except TypeError:
                    print(f'\n   ERROR! TypeError with parameter "{paramName}"...\n')
                    return -1
        try:
            if self.M == 0 or self.N == 0 or min(self.ParameterDict['K']) == 0 or \
                min(self.ParameterDict['MU']) == 0 or min(self.ParameterDict['CS']) == 0:
                raise ValueError
            self.ParameterDict['K']  = np.array(self.ParameterDict['K']).reshape(self.M)
            self.ParameterDict['Q']  = np.array(self.ParameterDict['Q']).reshape(self.M, self.M)
            self.ParameterDict['MU'] = np.array(self.ParameterDict['MU']).reshape(self.M)
            self.ParameterDict['CS'] = np.array(self.ParameterDict['CS']).reshape(self.M)
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
            value = self.ParameterDict['W'][i] / (self.ParameterDict['K'][i] * self.ParameterDict['MU'][i])
            if value > max:
                max = value
                indexMax = i
        return indexMax

    # Поиск lambdaArrij
    def find_lambdaArrij(self, lambdaArrayi):
        lambdaArrij = []
        for i in range(self.M):
            lambdaArrij.append([lambdaArrayi[i] * self.ParameterDict['Q'][i][j] for j in range(self.M)])
        return lambdaArrij

    # Расчёт матрицы A
    def solve_A(self, Wi, lambdaArrayi, lambdaArrij, Pi):
        A = []
        for i in range(self.M):
            A.append([Wi[i] * lambdaArrij[j][i] / lambdaArrayi[i] * self.ParameterDict['Q'][j][i] * (1 - Pi[j] ** 2) 
                      for j in range(self.M)])
            A[i][i] -= 1
        return A

    # Расчёт матрицы B
    def solve_B(self, Wi, lambdaArrayi, lambdaArrij, Xi, Pi):
        B = []
        for i in range(self.M):
            B.append(-1 + Wi[i] - Wi[i] * sum([lambdaArrij[j][i] / lambdaArrayi[i] * (1 - self.ParameterDict['Q'][j][i] + \
                self.ParameterDict['Q'][j][i] * (Pi[j] ** 2) * Xi[j]) for j in range(self.M)]))
        return B

    # Поиск CAi
    def find_CAi(self, lambdaArrayi, lambdaArrij):
        Pi = [lambdaArrayi[i] / self.ParameterDict['MU'][i] for i in range(self.M)]
        Vi = []
        for i in range(self.M):
            Vi.append(1 / sum([(self.ParameterDict['Q'][j][i] / lambdaArrayi[i]) ** 2 for j in range(self.M)]))
        Wi = [1 / (1 + 4 * ((1 - Pi[i]) ** 2) * (Vi[i] - 1)) for i in range(self.M)]
        Xi = [1 + (max(self.ParameterDict['CS'][i], 0.2) - 1) / np.sqrt(self.ParameterDict['K'][i]) for i in range(self.M)]
        A = self.solve_A(Wi, lambdaArrayi, lambdaArrij, Pi)
        B = self.solve_B(Wi, lambdaArrayi, lambdaArrij, Xi, Pi)       
        CAi = solve(A, B)
        return (Pi, CAi)

    # Поиск P0
    def find_P0(self, P, m):
        P0 = 1 + (P ** (m + 1)) / (self.ParameterDict['F'][m] * (m - P))
        for l in range(1, m + 1):
            P0 += (P ** l) / self.ParameterDict['F'][l]
        return 1 / P0

    # Поиск to без корректирующего фактора (toWF)
    def find_toWF(self, Pi, CAi):
        toWF = []
        for i in range(self.M):
            P = Pi[i]
            m = self.ParameterDict['K'][i]
            elem = (self.ParameterDict['CS'][i] + CAi[i]) / 2 * self.find_P0(P, m) * (P ** m) / (m * ((1 - P / m) ** 2) * \
                self.ParameterDict['F'][m] * self.ParameterDict['MU'][i])
            toWF.append(elem)
        return toWF

    # Выполнение одной итерации
    def doOneIter(self, Bi):
        lambdaArrayi = Bi * self.ParameterDict['W']
        lambdaArrij = self.find_lambdaArrij(lambdaArrayi)
        Pi, CAi = self.find_CAi(lambdaArrayi, lambdaArrij)
        toWF = self.find_toWF(Pi, CAi)
        to = [toWF[i] * (self.N - 1) / (self.N + toWF[i] * self.ParameterDict['MU'][i]) for i in range(self.M)]
        t = [to[i] + 1 / self.ParameterDict['MU'][i] for i in range(self.M)]
        jArray = [t[i] * lambdaArrayi[i] for i in range(self.M)]
        return (lambdaArrayi, jArray, t, to)

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
        Bi = self.ParameterDict['K'][indexMax] * (1 - 1 / self.N) * \
            self.ParameterDict['MU'][indexMax] / self.ParameterDict['W'][indexMax]
        self.xArray.append(Bi)
        self.yArray.append(Bi)
        L = 0.
        while abs(L - self.N) >= self.ErrorRate:
            iter = iter + 1
            lambdaArrayi, jArray, t, to = self.doOneIter(Bi)
            jArray = np.transpose(np.array(jArray))
            L = sum(jArray)
            Bi = Bi * self.N / L
            self.yArray.append(Bi)
            if iter > 0:
                Bi = self.funWegstein(iter)
            self.xArray.append(Bi)
        return (lambdaArrayi, jArray, t, to)

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
    def find_All(self, lambdaArray, jArray, t, to):
        no = [jArray[i] * to[i] / t[i] for i in range(self.M)]
        V = [self.N / lambdaArray[i] for i in range(self.M)]
        fi = self.M / sum(V)
        self.printResult(lambdaArray, jArray, t, V, no, to, fi)
        return

    # Главная функция
    def main(self, inputFileName):
        if self.splitInputFile(inputFileName) == -1:
            return 1
        lambdaArray, jArray, t, to = self.forIter(self.findMostLoadedNode())
        t = np.transpose(np.array(t))
        self.find_All(np.transpose(lambdaArray), jArray, t, to)
        return 0

if __name__ == '__main__':
    GMS_CF_GGK(FileName)