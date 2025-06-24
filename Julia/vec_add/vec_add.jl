using AMDGPU

function vadd!(c, a, b)
   i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
   if i <=  length(a)
       c[i] = a[i] + b[i]
   end
   return
end

n = 1024;
a = fill(1.0, n);
b = fill(2.0, n);
c = a .+ b;

a_d=ROCArray(a);
b_d=ROCArray(b);
c_d=a_d .+ b_d

c_h=Array(c_d);

error_check = 0

if c == c_h
   error_check += 1
end

fill!(c_d, 0.0);
groupsize = 256
gridsize = cld(n, groupsize)
@roc groupsize=groupsize gridsize=gridsize vadd!(c_d, a_d, b_d)

c_h = fill(0.0, n);
c_h=Array(c_d);

if c == c_h
   error_check += 1 
end

if  error_check == 2
    println("PASS!")
else

