# Exercise 1

## 1. computational intensity of LU decomposition

```c
// initialize all entries up to N
void ludecomp (int n, double A[])
{
  // transformation to upper triangular form
  for (std::size_t k=0; k<n-1; ++k)
    for (std::size_t i=k+1; i<n; ++i)
      {
        double q = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
        A[INDEX(i,k,n)] = q; // L matrix
        for (std::size_t j=k+1; j<n; ++j)
          A[INDEX(i,j,n)] -= q * A[INDEX(k,j,n)]; // rest of A matrix
      }
}

```

Number of FLOPs: 

$$
\begin{align}
\Sigma_{k=0}^{n-2} \Sigma_{i=k+1}^{n-1} (1+ \Sigma_{j=k+1}^{n-1} 2) = \frac{2n^3}{3}-\frac{n^2}{2} - \frac{n}{6} =: f_1(n)
\end{align}
$$
Data movement:
$$
\begin{align}\Sigma_{k=0}^{n-2} \Sigma_{i=k+1}^{n-1} (3*8+ \Sigma_{j=k+1}^{n-1} 3*8) =  8n^3 - 8n  =: f_2(n)
\end{align}
$$

$$
Intensity = \frac{\frac{2n^3}{3}-\frac{n^2}{2} - \frac{n}{6}}{ \frac{16}{3}n^3 + 4n^2 - \frac{28}{3}n} \approx \frac{1}{8}
$$

## 2. LU decomposition with block

   ```c++
   // initialize all entries up to N
   void ludecomp_blocked (int n, double A[])
   {
     if (n%M!=0) exit(1);
   
     for (std::size_t K=0; K<n; K+=M) //iterate over blocks
       {
         // 1a) LU decomposition of upper left block
         for (std::size_t k=K; k<K+M-1; ++k)
           for (std::size_t i=k+1; i<K+M; ++i)
             {
               double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
               A[INDEX(i,k,n)] = lik;
               for (std::size_t j=k+1; j<K+M; ++j)
                 A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
             }
         
         // 1b) remaining blocks in first column 
         for (std::size_t k=K; k<K+M; ++k)
           for (std::size_t i=K+M; i<n; ++i)
             {
               double lik = A[INDEX(i,k,n)]/A[INDEX(k,k,n)];
               A[INDEX(i,k,n)] = lik;
               for (std::size_t j=k+1; j<K+M; ++j)
                 A[INDEX(i,j,n)] -= lik * A[INDEX(k,j,n)];
             }
   
         // 2) Solve for U_KJ
         for (std::size_t J=K+M; J<n; J+=M)
           for (std::size_t i=0; i<M; ++i)
             for (std::size_t k=0; k<i; ++k)
               for (std::size_t j=0; j<M; ++j)
                 A[INDEX(K+i,J+j,n)] -= A[INDEX(K+i,K+k,n)]*A[INDEX(K+k,J+j,n)];
                   
         // 3) update S
         for (std::size_t I=K+M; I<n; I+=M)
           for (std::size_t J=K+M; J<n; J+=M)
             for (std::size_t i=0; i<M; ++i)
               for (std::size_t j=0; j<M; ++j)
                 for (std::size_t k=0; k<M; ++k)
                   A[INDEX(I+i,J+j,n)] -= A[INDEX(I+i,K+k,n)]*A[INDEX(K+k,J+j,n)];
       }
   }
   ```

   In this implementation, the algorithm traverse the whole matrix (N by N) by each step focusing on a square matrix with size M by M at the diagonal position. In each step:

   - 1a) Do a regular LU decomposition on a small square M by M matrix(*Let's call it pivot matrix*) at the upper left of the remaining matrix. To save memory space, we store the coefficients that are used to do the elimination at the corresponding position in the lower triangle part of the matrix.
   - 1b) Eliminate all elements in the same column with pivot matrix and under the pivot matrix. To save memory space, we store the coefficients that are used to do the elimination at the corresponding position in the column. Notice that we need to update once for the first column, twice for the second column and so on.
   - 2) Update the elements in the same rows with pivot matrix that were not updated in 1a), using the coefficients stored in 1a). Update here means do the same row operations did in 1a) to the rest part of the row.
   - 3) Update the lower right part that didn't updated in 1a) 1b) and 2). Update here means do the same row operations did in 1b).

## 3. Intensity for blocked version

   Number of FLOPs:

- 1a): $\frac{N}{M}*f_1(M)$

- 1b): In total, 1b) needs to process $(N-M)*\frac{N}{M}*\frac{1}{2} = \frac{(N-M)N}{2M}$ rows, each with M columns (The  formula for summing an arithmetic series)

    In total:

    $\frac{(N-M)N}{2M}*(M+\frac{(1+M)M}{2}*2) = \frac{(M+2)(N-M)N}{2}$ 

- 2): 2) needs to process $\frac{(N-M)N}{2M}$ columns

    For each column, it needs $f_1(M)$ operations 

    In total: $\frac{(N-M)N}{2M} *f_1(M) $ 

- 3): The total number of columns that 3) needs to process is $\frac{(N-M)N}{2M}$ 

    For each row the FLOPs is $M*2$

    In total: $\frac{(N-M)N}{2M}* M*2 = (N-M)N$

- All in total:
    $$
    \begin{align}
    & \frac{N}{M}*f_1(M) + \frac{(M+2)(N-M)N}{2} + \frac{(N-M)N}{2M} *f_1(M) + (N-M)N \\ &= \frac{(M+4)N(N-M)}{2} + f_1(M)* \frac{N(-M+N+2)}{2M} \\
    &= -\frac{M^3}{3} + \frac{M^2N}{3} + \frac{5M^2}{12} + \frac{MN}{4} -\frac{29}{12}M + \frac{23N}{12} - \frac{1}{6}
    \end{align}
    $$
    

Data movement:

- 1a): $\frac{N}{M}*f_2(M)$

- 1b): The data movement when focusing on single pivot matrix, when 1b) process Y rows, and when calculating `lik`:

    $Y*\frac{(M+1)M}{2}*3*8 = Y(M+1)M*12$

    In total:

    $(N-M)*\frac{N}{M}*\frac{1}{2}*(M+1)M = \frac{(N-M)N(M+1)}{2}$ (The   formula for summing an arithmetic series)

- 2): same with 1b), $N^2-NM$ 

- 3): $\frac{\frac{N}{M}-1}{6}[(\frac{N}{M}-1)*M][(\frac{N}{M}-1)*M]= \frac{(\frac{N}{M}-1)^3M^2}{6} = \frac{(N-M)^3}{6M}$ (The formula for summing the squares of an equivariant series)