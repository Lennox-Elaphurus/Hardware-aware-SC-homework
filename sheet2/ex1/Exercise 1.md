# Exercise 1

1. computational intensity of LU decomposition

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
\Sigma_{k=0}^{n-2} \Sigma_{i=k+1}^{n-1} (1+ \Sigma_{j=k+1}^{n-1} 2) = \frac{1}{3}n^3 + \frac{3}{2}n^2 - \frac{23}{6}n +2
\end{align}
$$
Memory access times:
$$
\begin{align}8*\Sigma_{k=0}^{n-2} \Sigma_{i=k+1}^{n-1} (2+ \Sigma_{j=k+1}^{n-1} 2) =  \frac{8}{3}n^3 + 16n^2 - \frac{104}{3}n + 16
\end{align}
$$

$$
Intensity \approx \lim_{n -> +\infin}\frac{\frac{1}{3}n^3 + \frac{3}{2}n^2 - \frac{23}{6}n +2}{\frac{8}{3}n^3 + 16n^2 - \frac{104}{3}n + 16} = \frac{1}{8}
$$

2. LU decomposition with block

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

   

   

