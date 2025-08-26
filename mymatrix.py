# mymatrix.py
import math

class MatrixOps:
    def __init__(self, matrix):
        self.matrix = matrix


    def transpose(self):
        rows = len(self.matrix)
        cols = len(self.matrix[0])
        return [[self.matrix[i][j] for i in range(rows)] for j in range(cols)]

    def multiply(self, A, B):
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def compute_AtA(self):
        A_T = self.transpose()
        return self.multiply(A_T, self.matrix)

    def identity(self, n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def qr_decomposition(self, A):
        m = len(A)
        n = len(A[0])
        Q = [[0.0 for _ in range(n)] for _ in range(m)]
        R = [[0.0 for _ in range(n)] for _ in range(n)]

        for j in range(n):
            v = [A[i][j] for i in range(m)]
            for i in range(j):
                R[i][j] = sum(Q[k][i] * A[k][j] for k in range(m))
                v = [v[k] - R[i][j] * Q[k][i] for k in range(m)]
            norm_v = math.sqrt(sum(x * x for x in v))
            if norm_v == 0:
                R[j][j] = 0
                for i in range(m):
                    Q[i][j] = 0
            else:
                R[j][j] = norm_v
                for i in range(m):
                    Q[i][j] = v[i] / norm_v

        return Q, R

    def eigen_qr(self, max_iter=500, tol=1e-10):
        n = len(self.matrix)
        A = [[self.matrix[i][j] for j in range(n)] for i in range(n)]
        V = self.identity(n)

        for _ in range(max_iter):
            Q, R = self.qr_decomposition(A)
            A = self.multiply(R, Q)
            V = self.multiply(V, Q)

            off_diag = sum(abs(A[i][j]) for i in range(n) for j in range(n) if i != j)
            if off_diag < tol:
                break

        eigvals = [A[i][i] for i in range(n)]
        return eigvals, V
