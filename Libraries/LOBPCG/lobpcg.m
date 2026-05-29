function [X, lambda] = lobpcg_simplified(X, A, prec, tol, maxit)
% Initial orthonormalization
 %base 
%[X, ~] = qr(X, 0);
% modified: CGS2

[X, ~] = cgs2(X);


X_lock = [];
lambda_lock = [];

P = [];
k = size(X,2);
for it = 1:maxit

    %-----------------------------
    % Apply operator
    %-----------------------------
    AX = A * X;

    %-----------------------------
    % Rayleigh quotient
    %-----------------------------
    Lambda = X' * AX;
    lambda = diag(Lambda);

    %-----------------------------
    % Residuals
    %-----------------------------
    R = AX - X * Lambda;
    res = vecnorm(R);

    fprintf('it %8d  max residual = %.3e\n', it, max(res));
    for ii = 1:length(res)
    fprintf("res %d: %15.15e \n", ii, res(ii));
    end

    %-----------------------------
    % Lock converged eigenpairs
    %-----------------------------
    locked = res < tol;

    if any(locked)
        X_lock      = [X_lock, X(:, locked)];
        lambda_lock = [lambda_lock; lambda(locked)];
    end

    % Keep active block only
    X = X(:, ~locked);
    R = R(:, ~locked);
    lambda = lambda(~locked);

    if isempty(X)
        break
    end

    %-----------------------------
    % Preconditioned residual
    %-----------------------------
    W = prec(R);

    % Orthogonalize W,P vs locked
    if ~isempty(X_lock)
        W = W - X_lock * (X_lock' * W);
        P = P - X_lock * (X_lock' * P);
    end

    % Orthogonalize W vs X,P
    W = W - X * (X' * W);

    if ~isempty(P)
        W = W - P * (P' * W);
    end

    % Drop tiny W directions
    idxW = vecnorm(W) > 1e-12;
    W = W(:, idxW);

    % Drop tiny P directions
    idxP = vecnorm(P) > 1e-12;
    P = P(:, idxP);

    %-----------------------------
    % Trial subspace (NO QR!)
    %-----------------------------
    S = [X, W, P];

    AS = S' * (A * S);
    BS = S' * S;

    %-----------------------------
    % Generalized Rayleigh–Ritz
    %-----------------------------
    AS = (AS + AS')/2;
   [Y, D] = eig(AS, BS);
   % BS = (BS + BS')/2;
   % AS = (AS + AS')/2;
   % 
   % RRR = chol(BS);
   % C = RRR' \ AS / RRR;
   % 
   % [C, D] = eig(C);
   % Y = RRR \ C;
    [theta, idx] = sort(diag(D), 'ascend');

    k_act = size(X,2);
    Y = Y(:, idx(1:k_act));

    %-----------------------------
    % Partition Ritz vectors
    %-----------------------------
    kX = size(X,2);
    kW = size(W,2);
    kP = size(P,2);

    Yx = Y(1:kX, :);
    Yw = Y(kX+1:kX+kW, :);
    Yp = Y(kX+kW+1:end, :);

    %-----------------------------
    % Update X and P (TRUE LOBPCG)
    %-----------------------------
    if (isempty(P))
        Xnew = X * Yx + W * Yw;
    else
        Xnew = X * Yx + W * Yw + P * Yp;
    end
   [Xnew, ~] = cgs2(Xnew);
   %[Xnew, ~] = qr(Xnew, 0);
    if (isempty(P))
        Pnew = W * Yw;
    else
        Pnew = W * Yw + P * Yp;
    end

    % Stabilize P
    Pnew = Pnew - Xnew * (Xnew' * Pnew);
    %[Pnew, Rp] = qr(Pnew, 0);
    [Pnew, Rp] = cgs2(Pnew);
    %Pnew = Pnew(:, abs(diag(Rp)) > 1e-12);

    %-----------------------------
    % Assign
    %-----------------------------
    X = Xnew;
    P = Pnew;
    lambda = theta(1:k_act);

    if size(X_lock,2) == k
        break
    end
end

% Combine locked + active
X = [X_lock, X];
lambda = [lambda_lock; lambda];

% Sort final result
[lambda, idx] = sort(lambda, 'ascend');
X = X(:, idx);
end

% function [QV, R] = cgs2(V)
% c = size(V,2);
% Q(:,1) = V(:,1)/norm(V(:,1));
% R(1,1) = norm(V(:,1));
% for i=2:c
%     a1 = Q(:,1:i-1)'*V(:,i);
%     Q(:,i) = V(:,i) - Q(:,1:i-1)*a1;
%      a2 = Q(:,1:i-1)'*Q(:,i);
%     Q(:,i) = Q(:,i) - Q(:,1:i-1)*a2;
%     R(1:i-1,i) = a1+a2;
%     nrm = norm(Q(:,i));
%     Q(:,i) = Q(:,i)/nrm;
%     R(i,i) = nrm;
% end
% 
% end

function [V, R] = cgs2(V)
c = size(V,2);
V(:,1) = V(:,1)/norm(V(:,1));
R(1,1) = norm(V(:,1));
for i=2:c
    a1 = V(:,1:i-1)'*V(:,i);
    Q(:,i) = V(:,i) - V(:,1:i-1)*a1;
     a2 = V(:,1:i-1)'*V(:,i);
    V(:,i) = Q(:,i) - V(:,1:i-1)*a2;
    R(1:i-1,i) = a1+a2;
    nrm = norm(V(:,i));
    V(:,i) = V(:,i)/nrm;
    R(i,i) = nrm;
end

end
