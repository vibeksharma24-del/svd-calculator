# svd.py
import streamlit as st
from mymatrix import MatrixOps
import math

st.title("Step-by-Step SVD Calculator")

# Step 1: Get matrix dimensions
rows = st.number_input("Number of rows", min_value=1, step=1, value=2)
cols = st.number_input("Number of columns", min_value=1, step=1, value=2)

# Step 2: Get matrix values
matrix = []
st.write("Enter matrix values:")
for i in range(int(rows)):
    row = []
    for j in range(int(cols)):
        val = st.number_input(f"Element [{i+1},{j+1}]", value=0.0, key=f"{i}_{j}")
        row.append(val)
    matrix.append(row)

# Step 3: Compute SVD
if st.button("Compute SVD"):
    if not matrix:
        st.warning("Please enter matrix values first!")
    else:
        m = MatrixOps(matrix)

        st.subheader("Step 1: Original Matrix (A)")
        st.write(m.matrix)

        t = m.transpose()
        st.subheader("Step 2: Transpose of A (A^T)")
        st.write(t)

        AtA = m.compute_AtA()
        st.subheader("Step 3: Compute A^T * A")
        st.write(AtA)

        eigs, vecs = MatrixOps(AtA).eigen_qr(max_iter=500, tol=1e-10)
        st.subheader("Step 4: Eigenvalues of A^T * A")
        st.write(eigs)

        st.subheader("Step 5: Eigenvectors of A^T * A (V columns)")
        st.write(vecs)

        singular_values = [math.sqrt(abs(ev)) for ev in eigs]
        st.subheader("Step 6: Singular values (σ)")
        st.write(singular_values)

        U = []
        for i, sigma in enumerate(singular_values):
            if sigma != 0:
                Av = [sum(m.matrix[row][k] * vecs[k][i] for k in range(len(vecs))) for row in range(len(m.matrix))]
                u_col = [x / sigma for x in Av]
            else:
                u_col = [0.0 for _ in range(len(m.matrix))]
            U.append(u_col)
        U = [[U[j][i] for j in range(len(U))] for i in range(len(U[0]))]
        st.subheader("Step 7: U matrix")
        st.write(U)

        Sigma = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(min(rows, cols)):
            Sigma[i][i] = singular_values[i]
        st.subheader("Step 8: Sigma matrix")
        st.write(Sigma)

        V_T = [[vecs[j][i] for j in range(len(vecs))] for i in range(len(vecs[0]))]
        Sigma_VT = [[sum(Sigma[i][k] * V_T[k][j] for k in range(len(V_T))) for j in range(len(V_T[0]))] for i in range(len(Sigma))]
        A_reconstructed = [[sum(U[i][k] * Sigma_VT[k][j] for k in range(len(U[0]))) for j in range(len(Sigma_VT[0]))] for i in range(len(U))]
        st.subheader("Step 9: Reconstructed A = U * Sigma * V^T")
        st.write(A_reconstructed)

        st.success("All steps complete!")
