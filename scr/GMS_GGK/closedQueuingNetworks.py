
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