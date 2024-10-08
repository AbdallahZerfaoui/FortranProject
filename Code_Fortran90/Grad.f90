module Grad
	use mod_fonctions_instationnaire
	implicit none 
  
contains
	subroutine Grad_conjuge(A,U,F,n,Nx)
    implicit none
    integer, intent(in) :: n,Nx
	double precision,dimension(1:5,1:n), intent(in) :: A
	double precision,dimension(1:n), intent(inout) :: U
	double precision,dimension(1:n), intent(in) :: F
	double precision,dimension(1:n)::Gr,DIR,V,Uk,Grk
	double precision::alpha,p,epsi=1E-9
	integer::compt,l
	
	Gr=prodMV(A,U,n,Nx)-1.d0*F !matmul(A,U)-F!prodMV(A,U,n)-F
	DIR= -Gr!F!-1*Gr

	do while (sqrt(dot_product(Gr,Gr)/dot_product(F,F))>Epsi)
		V=prodMV(A,DIR,n,Nx)
		!do l=1, n
			!write(175,*) V(l)!, Gr(l), DIR(l)
		!end do
	
		alpha=dot_product(Gr,Gr)/dot_product(V,DIR)
	
		!write(175,*) alpha, dot_product(V,DIR)
	
		Uk=U
		U=U+alpha*DIR
	!print*, U(1)
    !%==========================
    !% MISE A JOUR DU GRADIENT %
    !%==========================
		Grk=Gr
		Gr=Gr+alpha*V
    !%==================================
    !% CALCUL DE LA NOUVELLE DIRECTION %
    !%==================================
		p = dot_product(Gr,Gr)/dot_product(Grk,Grk)
		DIR=-1*Gr+p*DIR
	
		compt=compt+1
	end do
	!print*, "je suis le compt", compt
	end subroutine Grad_conjuge
end module Grad
