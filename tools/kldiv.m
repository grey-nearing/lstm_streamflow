function D = kldiv(X,Y,Bw)

X = X(:);
Y = Y(:);

By = (min(min(X),min(Y))-1e-6):Bw:(max(max(X),max(Y))+1e-6+Bw);
Bx = By;

Im = find(any(isnan([X,Y]),2));
X(Im) = [];
Y(Im) = [];

N = length(X);
assert(N==length(Y))

Mx = length(Bx);
My = length(By);

Pxy = zeros(Mx,My);
for n = 1:N
    x = X(n);
    y = Y(n);
    y = find(y<=By,1,'first');
    if isempty(y); keyboard; end
    x = find(x<=Bx,1,'first');
    if isempty(x); keyboard; end
    Pxy(x,y) = Pxy(x,y) + 1;
end
Pxy = max(Pxy,0);
Pxy = Pxy/sum(Pxy(:));

Px  = squeeze(sum(Pxy,2));
Py  = squeeze(sum(Pxy,1));

Px = Px(:);
Py = Py(:);

if abs(sum(Px)-1)>1/N^2;  error('Px does not sum to 1');  end
if abs(sum(Py)-1)>1/N^2;  error('Py does not sum to 1');  end

Hx = -Px(Px>0)'*log(Px(Px>0));

D = -Px(Px>0 & Py>0)'*log(Py(Px>0 & Py>0)./Px(Px>0 & Py>0));

D = D/Hx;


