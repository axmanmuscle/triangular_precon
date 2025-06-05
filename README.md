## Triangular Preconditioning
This is a project about puting a nonsymmetric preconditioner onto something like a proximal point method.

## To Do
 - Write up some of the math in this readme
 - list of papers
 - write some code that works

For the code, start with:
 - proximal point (done)
 - then code for our weighted proximal point with only a diagonal matrix (done)
   - check to make sure these match when the matrix is alpha*Identity
 - then when you change the entries
 - then fully triangular


## Math
Let $\mathcal{H}$ be a Hilbert space. We are focusing on the monotone inclusion problem
$$0 \in Tx$$
where $T : \mathcal{H} \to 2^{\mathcal{H}}$ is a monotone operator (usually maximally monotone). Let $V$ be another operator on $\mathcal{H}$, then we can write the above inclusion as
$Vx \in Vx + Tx$
or
$Vx \in (V + T)x.$
The resolvent is 
If $V$ is invertible this is $x \in (I + V^{-1}T) 

Need to code up a nontrivial problem that proximal point will work for (where the solution isn't just zero) to test this and check what matrices work

## Test Problem
Let's do a least squares problem so we can do a nontrivial minimization just using proximal point.

Let $f(x) = \lVert Ax - b \rVert_2^2$. evidently $\nabla f(x) = 2A^T(Ax - b)$ so we could compare to gradient descent. The prox operator will be $\text{prox}_{\lambda f}(x) = \left(AA^T + \frac{1}{\lambda}I\right)^1\left(A^Tb + \frac{1}{\lambda}x\right)$. Code this up and gen a random A and b.

## 6/1
Started adding the test problems from Parente et al. which are useful problems for PPA. This is in wp_test2.m. Make sure that proximal point converges to something like a minimum and then you (maybe) have test problems

