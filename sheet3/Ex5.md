# Ex5

I tried to implement a 4 x 4 masses in 

```c++
// loop over masses j
for (int j = J; j < J + B - 4; j += 4)
```

I think by changing the step from 2 to 4, we compute W x 2 masses to W x 4 masses at each time.

But there's no significant difference in my test of performance.