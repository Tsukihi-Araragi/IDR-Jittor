import jittor as jt

A = jt.array((1., 2.0)).reshape(1, 2)
B = jt.array((1.0, 2.0, 3., 4.)).reshape(2, 2)
print("A：   ",A)
print("B：   ",B)

C = A @ B
print("C：   ",C)

sum = C.sum()
print("sum = ", sum)
dA, dB = jt.grad(C, [A, B])
print("dA, dB: ", dA, dB)
A.stop_grad()
dA, dB = jt.grad(sum * sum, [A, B])
print("dA, dB: ", dA, dB)