using Statistics
using QuantumCumulants
using ModelingToolkit, OrdinaryDiffEq
using Distributions
using Plots

function tensor_cavs(N)
    Hlist = [FockSpace(Symbol(:cavity,i)) for i=1:N]
    return ⊗(Hlist...)
end
N=12
h = tensor_cavs(N)
a(i) = Destroy(h, Symbol(:a, i), i)
a_dag(i) = Create(h, Symbol(:a, i), i)
@cnumbers Δ γ J
H = sum(-Δ*a_dag(i)*a(i) for i=1:N) - sum(J*a_dag(i)*a(i+1) + J*a_dag(i+1)*a(i) for i=1:N-1) - J*(a_dag(1)*a(N) + a_dag(N)*a(1))
D = [a(i) for i=1:N]
rates = [γ for i=1:N]
H

#First order equations
eqs = meanfield([a(N)], H, D; rates=rates, order=1)
eqs_completed = complete(eqs)

#Numerical resolution of the mean-field equations
using ModelingToolkit, OrdinaryDiffEq
@named sys = ODESystem(eqs_completed)
p0 = (Δ=>1, J=>0.5, γ=>0.2)

list = [2, -1, 1im, 2im]
l = length(list)

u0 = zeros(ComplexF64, length(eqs_completed))

for i in 1:Int(3*l)
    num = mod(i, 4)
    if(num == 0)
        num=4
    end
    u0[i] = list[num]
end

prob = ODEProblem(sys,u0,(0.0,30.0),p0)
sol = solve(prob,Vern7(), dense=false, saveat = collect(range(0, 30,step=0.01))  )
print("ciao")

n = abs2.(sol[a(1)])
plot(sol.t, n, xlabel="t", ylabel="n", label="mean photon density cavity 1")

n = abs2.(sol[a(5)])
plot!(sol.t, n, xlabel="t", ylabel="n", label="mean photon density cavity 5")

n = abs2.(sol[a(10)])
plot!(sol.t, n, xlabel="t", ylabel="n", label="mean photon density cavity 10")

#Second order equations
eqs = meanfield([a(N)], H, D; rates=rates, order=2)
eqs_completed = complete(eqs)
