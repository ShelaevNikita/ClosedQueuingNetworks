
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './scr/MSM_MMK/InputData.dat'

class Jackson_Network_MMK():

    M = 1            # Количество узлов (приборов) в сети
    N = 1            # Количество заявок в сети
    ErrorRate = 0.1  # Погрешность выполнения программы

    InputParameterDict = { 'Q' : [],              # Матрица передач, определяющая маршрутизацию заявок в сети (размер - M x M)
                           'K' : np.empty(1),     # Число каналов обслуживания в узлах сети (размер - 1 x M)
                           'MU': np.empty(1),     # Интенсивность обслуживания заявок в узлах сети (размер - M)                          
                           'W' : np.empty(1),     # Вероятность нахождения заявки в текущем узле сети (размер - M)
                           'F' : np.empty(1) }    # Массив значений факториалов
    
    # 2 Массива для метода Вегстейна
    xArray = []
    yArray = []

    def __init__(self, inputFileName):
        self.main(inputFileName)

    # Подсчёт факториала заданного числа
    def factorial(self, n):
        if n <= 1:
            return 1
        else:
            return n * self.factorial(n - 1)

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
                        self.InputParameterDict[paramName] = np.array(list(map(int, paramArray)))
                    elif paramName == 'MU':
                        self.InputParameterDict[paramName] = np.array(list(map(float, paramArray)))
                    elif flagQ == True:
                        self.InputParameterDict['Q'] = self.InputParameterDict['Q'] + list(map(float, paramArray))
                        if len(self.InputParameterDict['Q']) == self.M ** 2:
                            flagQ = False
                except TypeError:
                    print(f'\n   Error! TypeError with parameter "{paramName}"...')
                    continue
        self.InputParameterDict['Q'] = np.array(self.InputParameterDict['Q']).reshape(self.M, self.M)
        return

    # Открытие файла с заданными параметрами сети
    def splitInputFile(self, inputFileName):
        try:
            inputFile = open(inputFileName, 'r', encoding = 'utf-8')
            self.getInputParameter(inputFile)
            inputFile.close()
        except FileNotFoundError:
            print(f'\n   ERROR! Requested file "{inputFileName}" not found!\n')
            return None
        return

    # Поиск массива W
    def findWArray(self):
        flag = False
        for i in range(self.M):
            for j in range(self.M):
                elem = self.InputParameterDict['Q'][i][j]
                if elem != 1 and elem != 0:
                    flag = True
                    break
            if flag:
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
            value = self.InputParameterDict['W'][i] / (self.InputParameterDict['MU'][i] * self.InputParameterDict['K'][i])
            if value > max:
                max = value
                indexMax = i
        return indexMax

    # Поиск Ji
    def findJi(self, lambdai, i):
        mui = self.InputParameterDict['MU'][i]
        k = self.InputParameterDict['K'][i]
        p = lambdai / mui
        sumH = sum([(p ** j) * 1 / self.InputParameterDict['F'][j] for j in range(1, (k + 1))])
        P0 = 1 / (1 + sumH + (p ** (k + 1)) / (self.InputParameterDict['F'][k] * (k - p)))
        toi = (p ** k) * P0 / (k * self.InputParameterDict['F'][k] * mui * ((1 - p / k) ** 2))
        hh = self.N / (self.N + toi * mui) * (self.N - 1) / self.N * toi
        Ji = lambdai * (hh + 1 / mui)
        return Ji

    # Выполнение одной итерации
    def doOneIter(self, Bi):
        lambdaArray = [Bi * elem for elem in self.InputParameterDict['W']]
        jArray = []
        for i in range(self.M): 
            jArray.append(self.findJi(lambdaArray[i], i))
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
        maxFact = max(self.M, max(self.InputParameterDict['K'])) + 1
        self.InputParameterDict['F'] = [self.factorial(j) for j in range(maxFact)]
        Bi = (self.InputParameterDict['K'][indexMax] * \
            self.InputParameterDict['MU'][indexMax]) / self.InputParameterDict['W'][indexMax]
        Bi = Bi * (1 - 1 / self.N)
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
    def printResult(self, lambdaArray, jArray):
        if len(lambdaArray) != self.M or len(jArray) != self.M:
            print('\n   Ошибка! Некорректные входные данные!')
            return
        t = [jArray[i] / lambdaArray[i] for i in range(self.M)]
        tWaiting = [t[i] - 1 / self.InputParameterDict['MU'][i] for i in range(self.M)]
        print('\n   lambda =', lambdaArray)
        print('\n   J =', jArray)
        print('\n   t =', t)
        print('\n   to =', tWaiting, '\n')
        return

    # Главная функция
    def main(self, inputFileName):
        self.splitInputFile(inputFileName)
        lambdaArray, jArray = self.forIter(self.findMostLoadedNode())
        self.printResult(lambdaArray, jArray)
        return 0

if __name__ == '__main__':
    Jackson_Network_MMK(FileName)
