
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './src/GSS_CF_GG1/InputData.dat'

class GSS_CF_GG1():

    M = 1            # Количество узлов (приборов) в сети
    N = 1            # Количество заявок в сети
    ErrorRate = 0.1  # Погрешность выполнения программы

    InputParameterDict = { 'Q' : [],   # Матрица передач, определяющая маршрутизацию заявок в сети (размер - M x M)
                           'MU': [],   # Интенсивность обслуживания заявок в узлах сети (размер - M)                          
                           'CS': [],   # Квадраты коэффициентов вариации длительностей обслуживания заявок в узлах сети (размер - M)
                           'W' : [] }  # Вероятность нахождения заявки в текущем узле сети (размер - M)

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
                    elif paramName in ['CS', 'MU']:
                        self.InputParameterDict[paramName] = list(map(float, paramArray))
                    elif flagQ == True:
                        self.InputParameterDict['Q'].append(list(map(float, paramArray)))
                        if len(self.InputParameterDict['Q']) == self.M:
                            flagQ = False
                except TypeError:
                    print(f'\n   Error! TypeError with parameter "{paramName}"...')
                    continue
        return

    # Открытие файла с заданными параметрами сети
    def splitInputFile(self, inputFileName):
        try:
            inputFile = open(inputFileName, 'r', encoding = 'utf-8')
            self.getInputParameter(inputFile)
            inputFile.close()
        except FileNotFoundError:
            print(f'\n   ERROR! Requested file "{inputFileName}" not found!\n')
            return
        return

    # Поиск массива W
    def findWArray(self):
        flag = False
        for elem in self.InputParameterDict['Q']:
            if not 1. in elem:
                flag = True
                break
        if flag:
            A = np.copy(np.transpose(self.InputParameterDict['Q']))
            B = np.zeros(self.M)
            for i in range(self.M):
                A[i][i] = A[i][i] - 1
            A[-1] = np.ones(self.M)
            B[-1] = 1
            self.InputParameterDict['W'] = solve(A, B)
        else:
            self.InputParameterDict['W'] = np.ones(self.M)
        return

    # Поиск наиболее загруженного узла в сети
    def findMostLoadedNode(self):
        self.findWArray()
        max = 0.
        indexMax = 0
        for i in range(self.M):
            value = self.InputParameterDict['W'][i] / self.InputParameterDict['MU'][i]
            if value > max:
                max = value
                indexMax = i
        return indexMax

    # Поиск lambdaArrij
    def find_lambdaArrij(self, lambdaArrayi):
        lambdaArrij = []
        for i in range(self.M):
            lambdaArrij.append([lambdaArrayi[i] * self.InputParameterDict['Q'][i][j] for j in range(self.M)])
        return lambdaArrij

    # Расчёт матрицы A
    def solve_A(self, Wi, lambdaArrayi, lambdaArrij, Xi, Pi):
        A = []
        for i in range(self.M):
            A.append([Wi[i] * lambdaArrij[j][i] / lambdaArrayi[i] * self.InputParameterDict['Q'][j][i] * (1 - Pi[j] ** 2) 
                      for j in range(self.M)])
            A[i][i] -= 1
        return A

    # Расчёт матрицы B
    def solve_B(self, Wi, lambdaArrayi, lambdaArrij, Xi, Pi):
        B = []
        for i in range(self.M):
            B.append(-1 + Wi[i] - Wi[i] * sum([lambdaArrij[j][i] / lambdaArrayi[i] * (1 - self.InputParameterDict['Q'][j][i] + \
                self.InputParameterDict['Q'][j][i] * (Pi[j] ** 2) * self.InputParameterDict['CS'][j]) for j in range(self.M)]))
        return B

    # Поиск CAi
    def find_CAi(self, lambdaArrayi, lambdaArrij):
        Pi = [lambdaArrayi[i] / self.InputParameterDict['MU'][i] for i in range(self.M)]
        Vi = []
        for i in range(self.M):
            Vi.append(1 / sum([(self.InputParameterDict['Q'][j][i] / lambdaArrayi[i]) ** 2 for j in range(self.M)]))
        Wi = [1 / (1 + 4 * ((1 - Pi[i]) ** 2) * (Vi[i] - 1)) for i in range(self.M)]
        Xi = [max(self.InputParameterDict['CS'][i], 0.2) for i in range(self.M)]
        A = self.solve_A(Wi, lambdaArrayi, lambdaArrij, Xi, Pi)
        B = self.solve_B(Wi, lambdaArrayi, lambdaArrij, Xi, Pi)       
        CAi = solve(A, B)
        return (Pi, CAi)

    # Поиск to без корректирующего фактора (toWF)
    def find_toWF(self, Pi, CAi):
        toWF = []
        for i in range(self.M):
            P = Pi[i]
            CA = CAi[i]
            elem = P * (self.InputParameterDict['CS'][i] + CA) / (2 * self.InputParameterDict['MU'][i] * (1 - P))
            if CA < 1. - self.ErrorRate:
                elem *= np.exp((P - 1) * ((1 - CA) ** 2) / (1.5 * P * (self.InputParameterDict['CS'][i] + CA)))
            elif CA > 1. + self.ErrorRate:
                elem *= np.exp((P - 1) * (CA - 1) / (4 * self.InputParameterDict['CS'][i] + CA))
            toWF.append(elem)
        return toWF

    # Выполнение одной итерации
    def doOneIter(self, Bi):
        lambdaArrayi = Bi * self.InputParameterDict['W']
        lambdaArrij = self.find_lambdaArrij(lambdaArrayi)
        Pi, CAi = self.find_CAi(lambdaArrayi, lambdaArrij)
        toWF = self.find_toWF(Pi, CAi)
        to = [toWF[i] * (self.N - 1) / (self.N + toWF[i] * self.InputParameterDict['MU'][i]) for i in range(self.M)]
        t = [to[i] + 1 / self.InputParameterDict['MU'][i] for i in range(self.M)]
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
        Bi = self.InputParameterDict['MU'][indexMax] / self.InputParameterDict['W'][indexMax] * (1 - 1 / self.N)
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
        if len(lambdaArray) != self.M or len(jArray) != self.M:
            print('\n   Ошибка! Некорректные входные данные!')
            return
        no = [jArray[i] * to[i] / t[i] for i in range(self.M)]
        V = [self.N / lambdaArray[i] for i in range(self.M)]
        fi = self.M / sum(V)
        self.printResult(lambdaArray, jArray, t, V, no, to, fi)
        return

    # Главная функция
    def main(self, inputFileName):
        self.splitInputFile(inputFileName)
        lambdaArray, jArray, t, to = self.forIter(self.findMostLoadedNode())
        t = np.transpose(np.array(t))
        self.find_All(np.transpose(lambdaArray), jArray, t, to)
        return 0

if __name__ == '__main__':
    GSS_CF_GG1(FileName)