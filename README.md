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
$
0 \in Tx
$
where $T : \mathcal{H} \to 2^{\mathcal{H}}$ is a monotone operator (usually maximally monotone). Let $V$ be another operator on $\mathcal{H}$, then we can write the above inclusion as
$
Vx \in Vx + Tx
$
or
$
Vx \in (V + T)x.
$


Need to code up a nontrivial problem that proximal point will work for (where the solution isn't just zero) to test this and check what matrices work