
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './scr/MSM_CF_MM1/InputData.dat'

class Jackson_Network_MM1():

    M = 1   # Количество узлов (приборов) в сети
    R = 1   # Количество классов заявок
    ErrorRate = 0.1  # Погрешность выполнения программы

    InputParameterDict = { 'N' : np.empty(1),   # Количество заявок определённого класса (размер - 1 x R)
                           'Q' : [],            # Матрица передач, определяющая маршрутизацию заявок в сети (размер - R x M x M)
                           'MU': [],            # Интенсивность обслуживания заявок в узлах сети (размер - R x M)                          
                           'W' : [] }           # Вероятность нахождения заявки в текущем узле сети (размер - R x M)

    # 2 Массива для метода Вегстейна
    xArray = []
    yArray = []

    def __init__(self, inputFileName):
        self.main(inputFileName)

    # Получение из входного файла значений параметров сети
    def getInputParameter(self, inputFile):
        flagQ = False
        flagMU = False
        for line in inputFile:
            lineParamIndex = line.find('=')
            if lineParamIndex > -1 or flagQ == True or flagMU == True:
                if flagQ or flagMU:
                    lineParamIndex = 0
                lineParam  = sub('[^0-9\.]', ' ', line[lineParamIndex:])
                paramArray = sub(r'\s+', ' ', lineParam.strip()).split()
                paramName  = line[(lineParamIndex - 3):(lineParamIndex - 1)].strip()
                if paramName == 'Q':
                    flagQ = True
                elif paramName == 'MU':
                    flagMU = True
                try:
                    if   paramName == 'M':
                        self.M = int(paramArray[0])
                    elif paramName == 'R':
                        self.R = int(paramArray[0])
                    elif paramName == 'E':
                        self.ErrorRate = float(paramArray[0])
                    elif paramName == 'N':
                        self.InputParameterDict[paramName] = np.array(list(map(int, paramArray)))
                    elif flagMU == True:
                        self.InputParameterDict['MU'] = self.InputParameterDict['MU'] + list(map(float, paramArray))
                        if len(self.InputParameterDict['MU']) == self.M * self.R:
                            flagMU = False
                    elif flagQ == True:
                        self.InputParameterDict['Q'] = self.InputParameterDict['Q'] + list(map(float, paramArray))
                        if len(self.InputParameterDict['Q']) == (self.M ** 2) * self.R:
                            flagQ = False
                except TypeError:
                    print(f'\n   Error! TypeError with parameter "{paramName}"...')
                    continue
        self.InputParameterDict['MU'] = np.array(self.InputParameterDict['MU']).reshape(self.R, self.M)
        self.InputParameterDict['Q'] = np.array(self.InputParameterDict['Q']).reshape(self.R, self.M, self.M)
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
        for r in range(self.R):
            flag = False
            for i in range(self.M):
                for j in range(self.M):
                    elem = self.InputParameterDict['Q'][r][i][j]
                    if elem != 1 and elem != 0:
                        flag = True
                        break
                if flag:
                    break    
            if flag:
                A = np.copy(np.transpose(self.InputParameterDict['Q'][r]))
                B = np.zeros(self.M)
                for i in range(self.M):
                    A[i][i] = A[i][i] - 1
                A[-1] = np.ones(self.M)
                B[-1] = 1
                self.InputParameterDict['W'].append(solve(A, B))
            else:
                self.InputParameterDict['W'].append(np.ones(self.M))
        return

    # Поиск наиболее загруженного узла в сети
    def findMostLoadedNode(self):
        self.findWArray()
        indexMaxArray = []
        for r in range(self.R):
            max = 0.
            indexMax = 0
            for i in range(self.M):
                value = self.InputParameterDict['W'][r][i] / self.InputParameterDict['MU'][r][i]
                if value > max:
                    max = value
                    indexMax = i
            indexMaxArray.append(indexMax)
        return indexMaxArray

    # Выполнение одной итерации
    def doOneIter(self, Bi):
        lambdaArrayi = []
        for i in range(self.M):
            lambdaArrayi.append([Bi[r] * self.InputParameterDict['W'][r][i] for r in range(self.R)])
        lambdaArrayR = [sum(lambdaArrayi[i]) for i in range(self.M)]
        ttR = []
        for i in range(self.M):
            ttR.append(sum([1 / self.InputParameterDict['MU'][r][i] * lambdaArrayi[i][r] / lambdaArrayR[i] for r in range(self.R)]))
        N = sum(self.InputParameterDict['N'])
        to = []
        for i in range(self.M):
            lambdaR = lambdaArrayR[i]
            tti = ttR[i]
            kk = lambdaR * (tti ** 2) / (1 - lambdaR * tti)
            to.append(N * tti / (N * tti + kk) * kk * (N - 1) / N)
        jArray = []
        for i in range(self.M):       
            jArray.append([lambdaArrayi[i][r] * (to[i] + 1 / self.InputParameterDict['MU'][r][i]) for r in range(self.R)])
        return (to, lambdaArrayi, jArray)

    # Функция метода Вегстейна
    def funWegstein(self, iter):
        yk = self.yArray[iter:(iter + 2)]
        xk = self.xArray[(iter - 1):(iter + 1)]
        BiNew = []
        for r in range(self.R):
            BiNew.append(yk[1][r] - ((yk[1][r] - yk[0][r]) * (yk[1][r] - xk[1][r])) / 
                         (yk[1][r] - yk[0][r] - xk[1][r] + xk[0][r]))
        return BiNew

    # Выполнение всех итераций
    def forIter(self, indexMaxArray):
        iter = -1
        Bi = [self.InputParameterDict['MU'][r][indexMaxArray[r]] / self.InputParameterDict['W'][r][indexMaxArray[r]] * \
            (1 - 1 / self.InputParameterDict['N'][r]) for r in range(self.R)]
        self.xArray.append(Bi)
        self.yArray.append(Bi)
        errorIter = self.ErrorRate * self.R
        L = np.zeros(self.R)
        while sum([abs(L[r] - self.InputParameterDict['N'][r]) for r in range(self.R)]) >= errorIter:
            iter = iter + 1
            to, lambdaArray, jArray = self.doOneIter(Bi)
            jArray = np.transpose(np.array(jArray))
            L = [sum(jArray[r]) for r in range(self.R)]
            Bi = [Bi[r] * self.InputParameterDict['N'][r] / L[r] for r in range(self.R)]
            self.yArray.append(Bi)
            if iter > 0:
                Bi = self.funWegstein(iter)
            self.xArray.append(Bi)
        return (to, lambdaArray, jArray)

    # Отображение полученного результата
    def printResult(self, to, lambdaArray, jArray):
        if len(lambdaArray) != self.R or len(jArray) != self.R:
            print('\n   Ошибка! Некорректные входные данные!')
            return
        t = []
        for r in range(self.R):
            t.append([jArray[r][i] / lambdaArray[r][i] for i in range(self.M)])
        print('\n   lambda =', lambdaArray)
        print('\n   J =', jArray)
        print('\n   t =', t)
        print('\n   to =', to, '\n')
        return

    # Главная функция
    def main(self, inputFileName):
        self.splitInputFile(inputFileName)
        to, lambdaArray, jArray = self.forIter(self.findMostLoadedNode())
        self.printResult(to, np.transpose(lambdaArray), jArray)
        return 0

if __name__ == '__main__':
    Jackson_Network_MM1(FileName)
