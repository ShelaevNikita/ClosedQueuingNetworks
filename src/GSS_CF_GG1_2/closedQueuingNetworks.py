
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './src/GSS_CF_GG1_2/InputData.dat'

class ClosedQueuningNetworks():

    M = 1            # Количество узлов (приборов) в сети
    R = 1            # Количество классов заявок
    ErrorRate = 0.1  # Погрешность выполнения программы

    InputParameterDict = { 'N' : np.empty(1),   # Количество заявок определённого класса (размер - 1 x R)
                           'Q' : [],            # Матрица передач, определяющая маршрутизацию заявок в сети (размер - R x M x M)
                           'MU': [],            # Интенсивность обслуживания заявок в узлах сети (размер - R x M)                          
                           'CS': [],            # Квадраты коэффициентов вариации длительностей обслуживания заявок в узлах сети (размер - R x M)
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
        flagCS = False
        for line in inputFile:
            lineParamIndex = line.find('=')
            if lineParamIndex > -1 or flagQ == True or flagMU == True or flagCS == True:
                if flagQ or flagMU or flagCS:
                    lineParamIndex = 0
                lineParam  = sub('[^0-9\.]', ' ', line[lineParamIndex:])
                paramArray = sub(r'\s+', ' ', lineParam.strip()).split()
                paramName  = line[(lineParamIndex - 3):(lineParamIndex - 1)].strip()
                if paramName == 'Q':
                    flagQ = True
                elif paramName == 'MU':
                    flagMU = True
                elif paramName == 'CS':
                    flagCS = True
                try:
                    if   paramName == 'M':
                        self.M = int(paramArray[0])
                    elif paramName == 'R':
                        self.R = int(paramArray[0])
                    elif paramName == 'E':
                        self.ErrorRate = float(paramArray[0])
                    elif paramName == 'N':
                        self.InputParameterDict[paramName] = np.array(list(map(int, paramArray)))
                    elif flagCS == True:
                        self.InputParameterDict['CS'] = self.InputParameterDict['CS'] + list(map(float, paramArray))
                        if len(self.InputParameterDict['CS']) == self.M * self.R:
                            flagCS = False
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
        self.InputParameterDict['CS'] = np.array(self.InputParameterDict['CS']).reshape(self.R, self.M)
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

    # Поиск lambdaArrij
    def find_lambdaArrij(self, lambdaArrayi):
        lambdaArrij = []
        for r in range(self.R):
            newArrij = []
            for i in range(self.M):
                newArrij.append([lambdaArrayi[i][r] * self.InputParameterDict['Q'][r][i][j] for j in range(self.M)])
            lambdaArrij.append(newArrij)
        return lambdaArrij

    # Поиск Q
    def find_Q(self, lambdaArrij, lambdaArrayR):
        lambdaArrRij = lambdaArrij[0]
        for r in range(1, self.R):
            for i in range(self.M):
                for j in range(self.M):
                    lambdaArrRij[i][j] += lambdaArrij[r][i][j]
        Q = []
        for i in range(self.M):
            Q.append([lambdaArrRij[i][j] / lambdaArrayR[i] for j in range(self.M)])
        return Q

    # Расчёт матрицы A
    def solve_A(self, Wi, lambdaQ, lambdaArrayR, Q, Pi):
        A = []
        for i in range(self.M):
            A.append([Wi[i] * lambdaQ[j][i] / lambdaArrayR[i] * Q[j][i] * (1 - Pi[j] ** 2) for j in range(self.M)])
        for i in range(self.M):
            A[i][i] -= 1
        return A

    # Расчёт матрицы B
    def solve_B(self, Wi, lambdaQ, lambdaArrayR, Q, Pi, CSai):
        B = []
        for i in range(self.M):
            B.append(-1 + Wi[i] - Wi[i] * sum([lambdaQ[j][i] * (Q[j][i] * (Pi[j] ** 2) * CSai[j] + 1 - Q[j][i]) / lambdaArrayR[i] \
                for j in range(self.M)]))
        return B

    # Поиск CAi
    def find_CAi(self, Q, MU, lambdaArrayi, lambdaArrayR):
        lambdaQ = []
        for i in range(self.M):
            lambdaQ.append([lambdaArrayR[i] * Q[i][j] for j in range(self.M)])
        CSai = []
        for i in range(self.M):
            CSai.append(sum([lambdaArrayi[i][r] / lambdaArrayR[i] * (self.InputParameterDict['CS'][r][i] + 1) * \
                ((MU[i] / self.InputParameterDict['MU'][r][i]) ** 2) for r in range(self.R)]) - 1)
        Pi = [lambdaArrayR[i] / MU[i] for i in range(self.M)]
        Vi = []
        for i in range(self.M):
            Vi.append(1 / sum([Q[j][i] / (lambdaArrayR[j] ** 2) for j in range(self.M)]))
        Wi = [1 / (1 + 4 * ((1 - Pi[i]) ** 2) * (Vi[i] - 1)) for i in range(self.M)]
        A = self.solve_A(Wi, lambdaQ, lambdaArrayR, Q, Pi)
        B = self.solve_B(Wi, lambdaQ, lambdaArrayR, Q, Pi, CSai)       
        CAi = solve(A, B)
        return (Pi, CSai, CAi)

    # Поиск to без корректирующего фактора (toWF)
    def find_toWF(self, Pi, CSai, CAi, MU):
        toWF = []
        for i in range(self.M):
            pi = Pi[i]
            CA = CAi[i]
            elem = pi * (CSai[i] + CA) / (2 * MU[i] * (1 - pi))
            if CA < 1 - self.ErrorRate:
                elem *= np.exp(-(1 - pi) * ((1 - CA) ** 2) / (1.5 * pi * (CSai[i] + CA)))
            elif CA > 1 + self.ErrorRate:
                elem *= np.exp(-(1 - pi) * (CA - 1) / (4 * CSai[i] + CA))
            toWF.append(elem)
        return toWF

    # Поиск t
    def find_t(self, to):
        t = []
        for i in range(self.M):
            t.append([to[i] + 1 / self.InputParameterDict['MU'][r][i] for r in range(self.R)])
        return t

    # Поиск jArray
    def find_J(self, lambdaArrayi, t):
        jArray = []
        for i in range(self.M):       
            jArray.append([t[i][r] * lambdaArrayi[i][r] for r in range(self.R)])
        return jArray

    # Выполнение одной итерации
    def doOneIter(self, Bi):
        lambdaArrayi = []
        for i in range(self.M):
            lambdaArrayi.append([Bi[r] * self.InputParameterDict['W'][r][i] for r in range(self.R)])
        lambdaArrayR = [sum(lambdaArrayi[i]) for i in range(self.M)]
        lambdaArrij = self.find_lambdaArrij(lambdaArrayi)
        Q = self.find_Q(lambdaArrij, lambdaArrayR)
        ttR = []
        for i in range(self.M):
            ttR.append(sum([1 / self.InputParameterDict['MU'][r][i] * lambdaArrayi[i][r] / lambdaArrayR[i] for r in range(self.R)]))
        MU = [1 / ttR[i] for i in range(self.M)]
        Pi, CSai, CAi = self.find_CAi(Q, MU, lambdaArrayi, lambdaArrayR)
        toWF = self.find_toWF(Pi, CSai, CAi, MU)
        N = sum(self.InputParameterDict['N'])
        to = [toWF[i] * ((N - 1) / N) * (N / (N + toWF[i] * MU[i])) for i in range(self.M)]
        t = self.find_t(to)
        jArray = self.find_J(lambdaArrayi, t)
        return (lambdaArrayi, jArray, t, to, MU, Q, CSai)

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
            lambdaArrayi, jArray, t, to, MU, Q, CSai = self.doOneIter(Bi)
            jArray = np.transpose(np.array(jArray))
            L = [sum(jArray[r]) for r in range(self.R)]
            Bi = [Bi[r] * self.InputParameterDict['N'][r] / L[r] for r in range(self.R)]
            self.yArray.append(Bi)
            if iter > 0:
                Bi = self.funWegstein(iter)
            self.xArray.append(Bi)
        return (lambdaArrayi, jArray, t, to, MU, Q, CSai)

    # Поиск среднего времени одного цикла Viv для v-го класса заявок через i-й узел сети
    def find_V(self, lambdaArray):
        V = []
        for r in range(self.R):
            V.append([self.InputParameterDict['N'][r] / lambdaArray[r][i] for i in range(self.M)])
        return V

    # Поиск пропускной способности сети fi
    def find_fi(self, V):
        fi = [1 / V[r][1] for r in range(self.R)]
        return fi

    # Поиск no
    def find_no(self, jArray, t, to):
        no = []
        for i in range(self.M):
            no.append(sum([jArray[r][i] * to[i] / t[r][i] for r in range(self.R)]))
        return no

    # Отображение полученного результата
    def printResult(self, lambdaArray, jArray, t, V, no, to, MU, Q, CSai, fi):
        print('\n   lambda =', lambdaArray)
        print('\n   MU =', MU)
        print('\n   Q =', Q)
        print('\n   CSai =', CSai)
        print('\n   n =', jArray)
        print('\n   t =', t)
        print('\n   V =', V)
        print('\n   no =', no)
        print('\n   to =', to)
        print('\n   fi =', fi, '\n')
        return

    # Вычисление других характеристик сети
    def find_All(self, lambdaArray, jArray, t, to, MU, Q, CSai):
        if len(lambdaArray) != self.R or len(jArray) != self.R:
            print('\n   Ошибка! Некорректные входные данные!')
            return
        no = self.find_no(jArray, t, to)
        V = self.find_V(lambdaArray)
        fi = self.find_fi(V)
        self.printResult(lambdaArray, jArray, t, V, no, to, MU, Q, CSai, fi)
        return

    # Главная функция
    def main(self, inputFileName):
        self.splitInputFile(inputFileName)
        lambdaArray, jArray, t, to, MU, Q, CSai = self.forIter(self.findMostLoadedNode())
        t = np.transpose(np.array(t))
        self.find_All(np.transpose(lambdaArray), jArray, t, to, MU, Q, CSai)
        return 0

if __name__ == '__main__':
    ClosedQueuningNetworks(FileName)