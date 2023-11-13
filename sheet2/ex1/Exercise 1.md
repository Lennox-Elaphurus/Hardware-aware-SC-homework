# Exercise 1

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

