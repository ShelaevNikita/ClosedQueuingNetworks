
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './src/MSS_D_MM1/InputData.dat'

class Single_M_Hetero_Closed_Nets():

    M = 1            # Количество узлов (приборов) в сети
    R = 1            # Количество классов заявок
    ErrorRate = 0.1  # Погрешность выполнения программы

    InputParameterDict = { 'N' : np.empty(1),   # Количество заявок определённого класса (размер - 1 x R)
                           'Q' : [],            # Матрица передач, определяющая маршрутизацию заявок в сети (размер - R x M x M)
                           'MU': [],            # Интенсивность обслуживания заявок в узлах сети (размер - R x M)                          
                           'W' : [] }           # Вероятность нахождения заявки в текущем узле сети (размер - R x M)

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

    # Поиск начального значения t
    def findFirst_t(self, k):       
        t = []
        for r in range(self.R):
            t.append([k / self.InputParameterDict['MU'][r][i] for i in range(self.M)])
        return t

    # Поиск Q = {qiv}
    def find_Q(self, t):
        Q = []
        for r in range(self.R):
            sumW = sum([self.InputParameterDict['W'][r][i] * t[r][i] for i in range(self.M)])
            Q.append([self.InputParameterDict['W'][r][i] * t[r][i] / sumW for i in range(self.M)])
        return Q

    # Поиск to
    def find_t(self, lambdaArray, Q):
        new_t = []
        for r in range(self.R):
            new_tr = []
            for i in range(self.M):
                t_ir = 0
                for v in range(self.R):
                    nv = self.InputParameterDict['N'][v]
                    if v == r:
                        nv -= 1
                    t_ir += nv * Q[v][i] / self.InputParameterDict['MU'][v][i]
                t_ir += 1 / self.InputParameterDict['MU'][r][i]
                new_tr.append(t_ir)
            new_t.append(new_tr)
        return new_t

    # Сравнение 2-х t
    def findComparison_t(self, t_last, t):
        flag = False
        for r in range(self.R):
            for i in range(self.M):
                error = abs(t_last[r][i] - t[r][i])
                if error >= self.ErrorRate:
                    flag = True
                    break
            if flag:
                break
        return flag

    # Поиск Lambda
    def find_Lambda(self, t, Q):
        lambdaArray = []
        for r in range(self.R):
            lambdaArray.append([self.InputParameterDict['N'][r] * Q[r][i] / t[r][i] for i in range(self.M)])
        return lambdaArray

    # Все итерации поиска t
    def doIter(self):
        self.findWArray()
        t_last =  self.findFirst_t(1)
        t = self.findFirst_t(sum(self.InputParameterDict['N']))
        while self.findComparison_t(t_last, t):
             Q = self.find_Q(t)
             lambdaArray = self.find_Lambda(t, Q)
             t_last = t
             t = self.find_t(lambdaArray, Q)
        return (lambdaArray, t)

    # Поиск среднего времени одного цикла Viv для v-го класса заявок через i-й узел сети
    def find_V(self, lambdaArray):
        V = []
        for r in range(self.R):
            V.append([self.InputParameterDict['N'][r] / lambdaArray[r][i] for i in range(self.M)])
        return V

    # Поиск jArray
    def find_J(self, lambdaArray, t):
        jArray = []
        for r in range(self.R):
            jArray.append([lambdaArray[r][i] * t[r][i] for i in range(self.M)])
        return jArray

    # Поиск tor
    def find_tor(self, t):
        tor = []
        for r in range(self.R):
            tor.append([t[r][i] - 1 / self.InputParameterDict['MU'][r][i] for i in range(self.M)])
        return tor

    # Поиск no
    def find_no(self, jArray, t, tor):
        no = []
        for i in range(self.M):
            no.append(sum([jArray[r][i] * tor[r][i] / t[r][i] for r in range(self.R)]))
        return no

    # Функция для расчёта суммы по всем классам заявок
    def find_result(self, array):
        result = []
        for i in range(self.M):
            result.append(sum([array[r][i] for r in range(self.R)]))
        return result

    # Функция для расчёта средневзвешенного по всем классам заявок
    def find_resultiv(self, multiv, firstiv, secondi):
        result = []
        for i in range(self.M):
            result.append(sum([multiv[r][i] * firstiv[r][i] / secondi[i] for r in range(self.R)]))
        return result

    # Поиск пропускной способности сети fi
    def find_fi(self, V):
        fi = [1 / V[r][1] for r in range(self.R)]
        return fi

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
    def find_All(self, lambdaArray, t):
        tor = self.find_tor(t)
        jArray = self.find_J(lambdaArray, t)
        ni = self.find_result(jArray)
        no = self.find_no(jArray, t, tor)
        lambda1 = self.find_result(lambdaArray)
        to = self.find_resultiv(tor, jArray, ni)
        V = self.find_V(lambdaArray)
        fi = self.find_fi(V)
        self.printResult(lambdaArray, jArray, t, V, no, to, fi)
        return

    # Главная функция
    def main(self, inputFileName):
        self.splitInputFile(inputFileName)        
        lambdaArray, t = self.doIter()
        self.find_All(lambdaArray, t)
        return 0

if __name__ == '__main__':
    Single_M_Hetero_Closed_Nets(FileName)
