
import numpy as np

from re import sub
from scipy.linalg import solve

FileName = './src/GMM_CF_GGK/InputData.dat'

class GMM_CF_GGK():

    M = 1            # Количество узлов (приборов) в сети
    R = 1            # Количество классов заявок
    ErrorRate = 0.1  # Погрешность выполнения программы

    ParameterDict = { 'N' : np.empty(1),  # Количество заявок определённого класса (размер - R)
                      'Q' : [],           # Матрица передач, определяющая маршрутизацию заявок в сети (размер - R x M x M)
                      'K' : np.empty(1),  # Число каналов обслуживания в узлах сети (размер - M)
                      'MU': [],           # Интенсивность обслуживания заявок в узлах сети (размер - R x M)                          
                      'CS': [],           # Квадраты коэффициентов вариации длительностей обслуживания заявок в узлах сети (размер - R x M)
                      'W' : [],           # Вероятность нахождения заявки в текущем узле сети (размер - R x M)
                      'F' : [] }          # Массив значений факториалов (размер - M)

    # 2 Массива для метода Вегстейна
    xArray = []
    yArray = []

    def __init__(self, inputFileName, flagStart = True):
        if flagStart:
            self.main(inputFileName)
    
    def returnParameterDict(self):
        res = self.ParameterDict.copy()
        res['M'] = self.M
        res['R'] = self.R
        return res

    # Получение из входного файла значений параметров сети
    def getInputParameter(self, inputFile):
        flagQ  = False
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
                if   paramName == 'Q':
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
                    elif paramName in ['N', 'K']:
                        self.ParameterDict[paramName] = np.array(list(map(int, paramArray)))
                    elif flagCS == True:
                        self.ParameterDict['CS'].append(list(map(float, paramArray)))
                        if len(self.ParameterDict['CS']) == self.R:
                            flagCS = False
                    elif flagMU == True:
                        self.ParameterDict['MU'].append(list(map(float, paramArray)))
                        if len(self.ParameterDict['MU']) == self.R:
                            flagMU = False
                    elif flagQ == True:
                        self.ParameterDict['Q'].append(list(map(float, paramArray)))
                        if len(self.ParameterDict['Q']) == self.M * self.R:
                            flagQ = False
                except TypeError:
                    print(f'\n   Error! TypeError with parameter "{paramName}"...')
                    continue
        self.ParameterDict['Q'] = np.array(self.ParameterDict['Q']).reshape(self.R, self.M, self.M)
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
        for r in range(self.R):
            flag = False
            for elem in self.ParameterDict['Q'][r]:
                if not 1. in elem:
                    flag = True
                    break    
            if flag:
                A = np.copy(np.transpose(self.ParameterDict['Q'][r]))
                B = np.zeros(self.M)
                for i in range(self.M):
                    A[i][i] = A[i][i] - 1
                A[-1] = np.ones(self.M)
                B[-1] = 1
                self.ParameterDict['W'].append(solve(A, B))
            else:
                self.ParameterDict['W'].append(np.ones(self.M))
        return

    # Поиск наиболее загруженного узла в сети
    def findMostLoadedNode(self):
        self.findWArray()
        indexMaxArray = []
        for r in range(self.R):
            max = 0.
            indexMax = 0
            for i in range(self.M):
                value = self.ParameterDict['W'][r][i] / (self.ParameterDict['K'][i] * self.ParameterDict['MU'][r][i])
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
                newArrij.append([lambdaArrayi[i][r] * self.ParameterDict['Q'][r][i][j] for j in range(self.M)])
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
            A[i][i] -= 1
        return A

    # Расчёт матрицы B
    def solve_B(self, Wi, lambdaQ, lambdaArrayR, Q, Pi, Xi):
        B = []
        for i in range(self.M):
            B.append(-1 + Wi[i] - Wi[i] * sum([lambdaQ[j][i] / lambdaArrayR[i] * \
                (Q[j][i] * (Pi[j] ** 2) * Xi[j] + 1 - Q[j][i]) for j in range(self.M)]))
        return B

    # Поиск CAi
    def find_CAi(self, Q, MU, lambdaArrayi, lambdaArrayR):
        lambdaQ = []
        for i in range(self.M):
            lambdaQ.append([lambdaArrayR[i] * Q[i][j] for j in range(self.M)])
        CSai = []
        for i in range(self.M):
            CSai.append(sum([lambdaArrayi[i][r] / lambdaArrayR[i] * (self.ParameterDict['CS'][r][i] + 1) * \
                ((MU[i] / self.ParameterDict['MU'][r][i]) ** 2) for r in range(self.R)]) - 1)
        Pi = [lambdaArrayR[i] / MU[i] for i in range(self.M)]
        Vi = []
        for i in range(self.M):
            Vi.append(1 / sum([Q[j][i] / (lambdaArrayR[j] ** 2) for j in range(self.M)]))
        Wi = [1 / (1 + 4 * ((1 - Pi[i]) ** 2) * (Vi[i] - 1)) for i in range(self.M)]
        Xi = [1 + (max(CSai[i], 0.2) - 1) / np.sqrt(self.ParameterDict['K'][i]) for i in range(self.M)]
        A = self.solve_A(Wi, lambdaQ, lambdaArrayR, Q, Xi)
        B = self.solve_B(Wi, lambdaQ, lambdaArrayR, Q, Pi, Xi)
        CAi = solve(A, B)
        return (Pi, CSai, CAi)

    # Поиск P0
    def find_P0(self, P, m):
        P0 = 1 + (P ** (m + 1)) / (self.ParameterDict['F'][m] * (m - P))
        for l in range(1, m + 1):
            P0 += (P ** l) / self.ParameterDict['F'][l]
        return 1 / P0

    # Поиск to без корректирующего фактора (toWF)
    def find_toWF(self, Pi, CSi, CAi, MU):
        toWF = []
        for i in range(self.M):
            P = Pi[i]
            m = self.ParameterDict['K'][i]
            elem = (CSi[i] + CAi[i]) / 2 * self.find_P0(P, m) * (P ** m) / (m * ((1 - P / m) ** 2) * \
                self.ParameterDict['F'][m] * MU[i])
            toWF.append(elem)
        return toWF

    # Поиск t
    def find_t(self, to):
        t = []
        for i in range(self.M):
            t.append([to[i] + 1 / self.ParameterDict['MU'][r][i] for r in range(self.R)])
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
            lambdaArrayi.append([Bi[r] * self.ParameterDict['W'][r][i] for r in range(self.R)])
        lambdaArrayR = [sum(lambdaArrayi[i]) for i in range(self.M)]
        lambdaArrij = self.find_lambdaArrij(lambdaArrayi)
        Q = self.find_Q(lambdaArrij, lambdaArrayR)
        ttR = []
        for i in range(self.M):
            ttR.append(sum([1 / self.ParameterDict['MU'][r][i] * lambdaArrayi[i][r] / lambdaArrayR[i] for r in range(self.R)]))
        MU = [1 / ttR[i] for i in range(self.M)]
        Pi, CSi, CAi = self.find_CAi(Q, MU, lambdaArrayi, lambdaArrayR)
        toWF = self.find_toWF(Pi, CSi, CAi, MU)
        N = sum(self.ParameterDict['N'])
        to = [toWF[i] * (N - 1) / (N + toWF[i] * MU[i]) for i in range(self.M)]
        t = self.find_t(to)
        jArray = self.find_J(lambdaArrayi, t)
        return (lambdaArrayi, jArray, t, to, MU, Q, CSi)

    # Функция метода Вегстейна
    def funWegstein(self, iter):
        yk = self.yArray[iter:(iter + 2)]
        xk = self.xArray[(iter - 1):(iter + 1)]
        BiNew = []
        for r in range(self.R):
            BiNew.append(yk[1][r] - ((yk[1][r] - yk[0][r]) * (yk[1][r] - xk[1][r])) / 
                         (yk[1][r] - yk[0][r] - xk[1][r] + xk[0][r]))
        return BiNew

    def findFactorial(self):
         self.ParameterDict['F'] = [np.math.factorial(j) for j in range(max(self.ParameterDict['K']) + 1)]
         return

    # Выполнение всех итераций
    def forIter(self, indexMaxArray):
        iter = -1
        Bi = [self.ParameterDict['K'][indexMaxArray[r]] * (1 - 1 / self.ParameterDict['N'][r]) * \
            self.ParameterDict['MU'][r][indexMaxArray[r]] / self.ParameterDict['W'][r][indexMaxArray[r]] for r in range(self.R)]
        self.xArray.append(Bi)
        self.yArray.append(Bi)
        errorIter = self.ErrorRate * self.R
        L = np.zeros(self.R)
        while sum([abs(L[r] - self.ParameterDict['N'][r]) for r in range(self.R)]) >= errorIter:
            iter = iter + 1
            lambdaArrayi, jArray, t, to, MU, Q, CSi = self.doOneIter(Bi)
            jArray = np.transpose(np.array(jArray))
            L = [sum(jArray[r]) for r in range(self.R)]
            Bi = [Bi[r] * self.ParameterDict['N'][r] / L[r] for r in range(self.R)]
            self.yArray.append(Bi)
            if iter > 0:
                Bi = self.funWegstein(iter)
            self.xArray.append(Bi)
        return (lambdaArrayi, jArray, t, to, MU, Q, CSi)

    # Поиск среднего времени одного цикла Viv для v-го класса заявок через i-й узел сети
    def find_V(self, lambdaArray):
        V = []
        for r in range(self.R):
            V.append([self.ParameterDict['N'][r] / lambdaArray[r][i] for i in range(self.M)])
        return V

    # Поиск пропускной способности сети fi
    def find_fi(self, V):
        fi = [self.M / sum(V[r]) for r in range(self.R)]
        return fi

    # Поиск no
    def find_no(self, jArray, t, to):
        no = []
        for i in range(self.M):
            no.append(sum([jArray[r][i] * to[i] / t[r][i] for r in range(self.R)]))
        return no

    # Отображение полученного результата
    def printResult(self, lambdaArray, jArray, t, V, no, to, MU, Q, CSi, fi):
        print('\n   lambda =', lambdaArray)
        print('\n   MUa =', MU)
        print('\n   Qa =', Q)
        print('\n   CSa =', CSi)
        print('\n   n =', jArray)
        print('\n   t =', t)
        print('\n   V =', V)
        print('\n   no =', no)
        print('\n   to =', to)
        print('\n   fi =', fi, '\n')
        return

    # Вычисление других характеристик сети
    def find_All(self, lambdaArray, jArray, t, to, MU, Q, CSi):
        if len(lambdaArray) != self.R or len(jArray) != self.R:
            print('\n   Ошибка! Некорректные входные данные!')
            return
        no = self.find_no(jArray, t, to)
        V = self.find_V(lambdaArray)
        fi = self.find_fi(V)
        self.printResult(lambdaArray, jArray, t, V, no, to, MU, Q, CSi, fi)
        return

    # Главная функция
    def main(self, inputFileName):
        self.splitInputFile(inputFileName)
        self.findFactorial()
        lambdaArray, jArray, t, to, MU, Q, CSi = self.forIter(self.findMostLoadedNode())
        t = np.transpose(np.array(t))
        self.find_All(np.transpose(lambdaArray), jArray, t, to, MU, Q, CSi)
        return 0

if __name__ == '__main__':
    GMM_CF_GGK(FileName)