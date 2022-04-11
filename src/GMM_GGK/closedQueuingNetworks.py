
import numpy as np

from re import sub

FileName = './InputData.dat'

ErrorRate = 0.1

class ClosedQueuningNetworks():

    M = 1   # Количество узлов (приборов) в сети
    R = 1   # Количество классов заявок

    InputParameterDict = { 'N' : np.empty(1),                   # Количество заявок определённого класса (размер - 1 x R)
                           'K' : np.empty(1),                   # Число каналов обслуживания в узлах сети (размер - 1 x M)
                           'Q' : np.empty(1).reshape(1, 1, 1),  # Матрица передач, определяющая маршрутизацию заявок в сети (размер - R x M x M)
                           'MU': np.empty(1).reshape(1, 1),     # Интенсивность обслуживания заявок в узлах сети (размер - R x M)                          
                           'CS': np.empty(1).reshape(1, 1),     # Квадраты коэффициентов вариации длительностей обслуживания заявок в узлах сети (размер - R x M)
                           'W' : np.empty(1).reshape(1, 1) }    # Вероятность нахождения заявки в текущем узле сети (размер - R x M)

    def __init__(self, inputFileName, errorRate):
        self.main(inputFileName, errorRate)
    
    # Подсчёт факториала заданного числа
    def factorial(self, n):
        if n <= 1:
            return 1
        else:
            return n * self.factorial(n - 1)
    
    # Подсчёт факториала числа каналов
    def factK(self):
        factK = []
        for m in range(self.M):
            factK.append(self.factorial(self.InputParameterDict['K'][m]))
        return np.array(factK)

    # Получение из входного файла значений параметров сети
    def getInputParameter(self, inputFile):
        for line in inputFile:
            lineParamIndex = line.find('=')
            if lineParamIndex > -1:
                lineParam  = sub('[^0-9\.]', ' ', line[lineParamIndex:])
                paramArray = sub(r'\s+', ' ', lineParam.strip()).split()
                paramName  = line[(lineParamIndex - 3):(lineParamIndex - 1)].strip()
                try:
                    if   paramName == 'M':
                        self.M = int(paramArray[0])
                    elif paramName == 'R':
                        self.R = int(paramArray[0])
                    elif paramName in ['N', 'K']:
                        self.InputParameterDict[paramName] = np.array(list(map(int, paramArray)))
                    else:
                        self.InputParameterDict[paramName] = np.array(list(map(float, paramArray)))
                except TypeError:
                    print(f'\n   Error! TypeError with parameter "{paramName}"...')
                    continue
        self.InputParameterDict['Q'] = self.InputParameterDict['Q'].reshape(self.R, self.M, self.M)
        for key in ['MU', 'CS']:
            self.InputParameterDict[key] = self.InputParameterDict[key].reshape(self.R, self.M)
        return

    # Открытие файла с заданными параметрами сети
    def splitInputFile(self, inputFileName):
        try:
            inputFile = open(inputFileName, 'r', encoding = 'utf-8')
            self.getInputParameter(inputFile)
            inputFile.close()
        except FileNotFoundError:
            print(f'\n   ERROR! Requested file "{filename}" not found!')
            return None
        return

    # Поиск наиболее загруженного узла в сети
    def findMostLoadedNode(self):
        indexLoadedNode = []
        for k in range(self.R):
            max = 0.0
            indexMax = 0
            for m in range(self.M):
                value = self.InputParameterDict['W'][k][m] / self.InputParameterDict['MU'][k][m]
                if value > max:
                    max = value
                    indexMax = m
            indexLoadedNode.append(indexMax)
        return np.array(indexLoadedNode)

    # Расчёт начального значения B
    def startValueB(self):
        indexLoadedNode = self.findMostLoadedNode()
        arrayB = []
        for k in range(self.R):
            index = indexLoadedNode[k]
            valueB = self.InputParameterDict['MU'][k][index] / self.InputParameterDict['W'][k][index] * (1 - 1 / self.InputParameterDict['N'][k])
            arrayB.append(valueB)
        return np.array(arrayB) 

    # Расчёт значений lambdaA
    def getLambdaA(self, arrayB):
        lambdaI = np.empty(self.R)
        lambdaA = []
        for k in range(self.R):
            lambdaI[k] = self.InputParameterDict['W'][k] * arrayB[k]
        for m in range(self.M):
            lambdaA.append(sum(lambdaI[0:, m]))
        return (lambdaI, np.array(lambdaA))
    
    # Расчёт значений muA
    def getArrayMuA(self, lambdaI, lambdaA):
        muA = []
        for m in range(self.M):
            value = 0.0
            for k in range(self.R):
                value += (1 / self.InputParameterDict['MU'][k][m]) * (lambdaI[k][m] / lambdaA[m])
            muA.append(1 / value)
        return np.array(muA)

    # Расчёт значений CSa
    def gettArrayCSa(self, lambdaI, lambdaA, muA):
        CSa = []
        for m in range(self.M):
            value = 0.0
            for k in range(self.R):
                value += (lambdaI[k][m] / lambdaA[m]) * (self.InputParameterDict['CS'][k][m] + 1) * \
                    ((muA[m] / self.InputParameterDict['MU'][k][m]) ** 2) - 1
            CSa.append(value)
        return np.array(CSa)

    def main(self, inputFileName):
        self.splitInputFile(inputFileName)
        factK = self.factK()
        iter = 0
        arrayB = self.startValueB()
        lambdaI, lambdaA = self.getLambdaA(arrayB)
        muA = self.getArrayMuA(lambdaI, lambdaA)
        CSa = self.gettArrayCSa(lambdaI, lambdaA, muA)
        return 0

if __name__ == '__main__':
    ClosedQueuningNetworks(FileName, ErrorRate)