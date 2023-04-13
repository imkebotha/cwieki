function C = cross_cov(A,B)

    mA = mean(A);
    mB = mean(B);
    
    [J,nA] = size(A);
    [J,nB] = size(B);
    
    C = zeros(nA,nB);
    for j = 1:J
        C = C + ((A(j,:) - mA)')*(B(j,:) - mB);
    end
    C = C/(J-1);

end