function Barnardextest(a,b,c,d)
%BARNARDEXTEST Barnard's Exact Probability Test.
%   BARNARDEXTEST, as the Fisher's exact test, performs the exact probability
%   test for a table of frequency data cross-classified according to two categorical
%   variables, each of which has two levels or subcategories (2x2). It is a 
%   non-parametric statistical test used to determine if there are nonrandom 
%   associations between the two categorical variables. Barnard's exact test is used
%   to calculate an exact P-value with small number of expected frequencies, for
%   which the Chi-square test is not appropriate (in case the total number of
%   observations is less than 20 or the number of frequency cells are less than
%   5). The test was proposed by G. A. Barnard in two papers (1945 and 1947).
%   While Barnard's test seems like a natural test to consider, it's not at all
%   commonly used. This probably due that it is a little unknown.
%
%   According to the next 2x2 table design,
%
%                    Var.1
%                --------------
%                  a       b      r1=a+b
%         Var.2
%                  c       d      r2=c+d
%                --------------
%                c1=a+c  c2=b+d  n=c1+c2
%
%   The Barnard's exact test is a unconditioned test for it generates the exact
%   distribution of the Wald statistic T(X),
%
%           T(X) = abs((p(a) - p(b))/sqrt(p*(1-p)*((1/c1)+(1/c2)))),
%   where,
%           p(a) = a/c1, p(b) = b/c2 and p = (a+b)/n, 
%
%   by considering all tables X � H and calculates P(PI) for all possible values of 
%   PI in (0, 1). The P-value is defined as the maximum value of p(PI),
%
%          p(PI) = SUM_T(X)>=T(Xo) C(c1,a)*C(c2,b)*PI^(a+b)*(1-PI)^(n-a-b).
%
%   Where T(Xo) is the actual (observed) Wald statistic.
%
%   We calculate P(PI) for all possible values of PI in (0, 1) and choose the
%   value that maximizes P(PI).
%
%   So, the Barnard's exact P-value is evaluated as,
%
%          P = sup{P(PI): PI � (0, 1)}.  
%
%   Perhaps due to its computational difficulty it is not widely used until recently,
%   where the computers make it feasible. It is considering that the Barnard's exact 
%   test is more powerful than the Fisher's one as a direct consequence of restricting
%   the Fisher's test to 2x2 tables belonging to the conditional reference H(a+b),
%   rather than the larger unconditional reference of Barnard's set for all H's.
%   Consequently, the number of distinct P-values obtained with the Fisher's exact
%   test is less than the corresponding number that one could get with the Barnard's exact
%   test. So, for a restrict alpha-error, the Fisher's procedure will usually be more
%   conservative, resulting in a loss of power.
%
%   Syntax: function Barnardextest(a,b,c,d) 
%      
%   Inputs:
%         a,b,c,d - observed frequency cells
%
%   Output:
%         A table with:
%         - Wald statistic, Nuisance parameter and P-value
%         - Plot of the nuisance parameter PI against the corresponding P-value for
%           all the PI in (0, 1). It shows the maximized PI where it attains the
%           P-value. For a least conflict with figure, press the left mouse button on 
%           the legend and drag to the desired location.
%
%   Example: From the example given on the Ina Parks S. Howell's internet homepage 
%            (http://www.fiu.edu/~howellip/Fisher.pdf). Suppose Crane and Egret are
%            two very small collages, the results of the beginning physics course at
%            each of the two schools are given in the follow table.
%
%                                   Physics
%                              Pass         Fail
%                            ---------------------
%                      Crane     8           14
%            Collage                 
%                      Egret     1            3
%                            ---------------------
%                                       
%   Calling on Matlab the function: 
%             Barnardextest(8,14,1,3)
%
%   Answer is:
%
%   Table of the Barnard's exact test results for:
%   a = 8, b = 14, c = 1, d = 3
%   -------------------------------------------------
%     Wald statistic     Nuisance parameter      P  
%   -------------------------------------------------
%         0.4394               0.9001          0.4159
%   -------------------------------------------------
%
%   [Note:For a least conflict with figure, press the left mouse button on the legend and drag to the desired location.]
%
%  Created by A. Trujillo-Ortiz, R. Hernandez-Walls and A. Castro-Perez
%             Facultad de Ciencias Marinas
%             Universidad Autonoma de Baja California
%             Apdo. Postal 453
%             Ensenada, Baja California
%             Mexico.
%             atrujo@uabc.mx
%             And the special collaboration of the post-graduate students of the 2004:2
%             Multivariate Statistics Course: Laura Rodriguez-Cardozo, Norma Alicia Ramos-Delgado,
%             and Rene Garcia-Sanchez. 
%  Copyright (C) October 25, 2004.
%
%  To cite this file, this would be an appropriate format:
%  Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez, L. Rodriguez-Cardozo 
%    N.A. Ramos-Delgado and R. Garcia-Sanchez. (2004). Barnardextest:Barnard's Exact
%    Probability Test. A MATLAB file. [WWW document]. URL http://www.mathworks.com/
%    matlabcentral/fileexchange/loadFile.do?objectId=6198
%
%  References:
% 
%  Barnard, G.A. (1945), A new test for 2x2 tables. Nature, 156:177.
%  Barnard, G.A. (1947), Significance tests for 2x2 tables. Biometrika, 34:123-138. 
%  Howell, I.P.S. (Internet homepage), http://www.fiu.edu/~howellip/Fisher.pdf
%  Mehta, C.R. and Hilton, J.F. (1993), Exact power of conditional and unconditional
%           tests: going beyond the 2x2 contingency table. American Statistician,
%           47:91-98.
%
if nargin < 4, 
    error('Too few arguments.'); 
end;
c1 = a+c;
c2 = b+d;
n = c1+c2;
pao = a/c1;
pbo = b/c2;
pxo = (a+b)/n;
warning off
TXO = abs(pao-pbo)/sqrt(pxo*(1-pxo)*((1/c1)+(1/c2)));
warning on
nans = isnan(TXO);
mo = find(nans);
TXO(mo) = zeros(size(mo));
n1 = prod(1:c1);
n2 = prod(1:c2);
P = [];
for p=.0001:.01:1
    S = [];TX = [];
    for i = 0:c1
        for j = 0:c2
            fac1 = prod(1:i);
            fac2 = prod(1:j);
            fac3 = prod(1:(c1-i));
            fac4 = prod(1:(c2-j));
            s = (n1*n2*(p.^(i+j))*(1-p).^(n-(i+j)))./(fac1*fac2*fac3*fac4);
            S = [S s];
            pa = i/c1;
            pb = j/c2;
            px = (i+j)/n;
            warning off
            tx = (pa-pb)/sqrt(px*(1-px)*((1/c1)+(1/c2)));
            warning on
            TX = [TX tx];
            nans = isnan(TX);
            m = find(nans);
            TX(m) = zeros(size(m));
        end;
    end;
    P = [P;sum(S(find(TX >= TXO)))];
end;
p = 0.0001:.01:1;
plot(p,P,'b-')
np = find(P==max(P));
[p(np) P(np)];
disp(' ')
disp('Table of the Barnard''s exact test results for:')
disp(['a = ' num2str(a) ', b = ' num2str(b) ', c = ' num2str(c) ', d = ' num2str(d) '']);
fprintf('---------------------------------------------------\n');
disp('  Wald statistic     Nuisance parameter      P  '); 
fprintf('---------------------------------------------------\n');
fprintf('  %10.4f           %10.4f      %10.4f\n',[TXO,p(np),P(np)].');
fprintf('---------------------------------------------------\n');
disp(' ')
disp('[Note:For a least conflict with figure, press the left mouse button on the legend and drag to the desired location.]')
hold on
plot(p(np),P(np),'ro')
title('Barnard''s exact P-value as a function of nuisance parameter PI.','FontSize',12);
xlabel('Nuisance parameter (PI)');
ylabel('P-value  [P( TX >= TXO | PI )]');
legend(['P-value = ',num2str(P(np))],['Nuisance parameter (PI) = ',num2str(p(np))]);
return;