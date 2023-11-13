# Exercise 1

1.

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
\Sigma_{k=0}^{n-2} \Sigma_{i=k+1}^{n-1} (1+ \Sigma_{j=k+1}^{n-1} 2) &= \Sigma_{k=0}^{n-2} \Sigma_{i=k+1}^{n-1} (1+2*(n-k-1))\\
&= \left\{
\begin{aligned}
& \frac{1}{2}n^3 + \frac{3}{2}n^2 - 2n , && \text{if $n$ is even} \\
 &\frac{1}{2}n^3 + \frac{5}{2}n^2 - \frac{7}{2}n - \frac{1}{2}, && \text{if $n$ is odd}
\end{aligned}
\right.
\end{align}
$$
