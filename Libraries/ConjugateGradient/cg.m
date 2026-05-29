function [x, iter, res] = CG_impl(A, b, x0, tol, max_it, precon)

r = b - A * x0;
x = x0;
res(1) = norm(r);

fprintf("it %d norm %5.5e \n", 0, res(1));

notconv = 1;
iter = 1;

while (notconv == 1)
    %   apply preconditioner to r, i.e., w = K^{-1}r
    
    rho_current = r'*w;

    if (iter == 1)
        p = w;
    else
        bet = rho_current/rho_previous;
        p = w + bet * p;
    end

    q = A * p;

    alph = rho_current / (p' * q);
    x = x + alph * p;
    r = r - alph * q;

    nrm_r = norm(r);
    fprintf("it %d norm %5.5e \n", iter, nrm_r);

    if (( nrm_r / res(1) < tol ) || ( iter>max_it ))
        notconv = 1;
    end
    res(iter+1) = nrm_r;
    iter = iter + 1;
    rho_previous = rho_current;
end
end


