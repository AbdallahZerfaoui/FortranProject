module mod_fonctions_instationnaire
  implicit none 
  
contains

!===================================================
!fonction f
!==========================================
  function f1(x,y,t)
    double precision:: x, y, f1, Lx, Ly,D, pi=acos(-1.0),t
    integer :: Nx, Ny
    
    !===================================================
	!Recuperation des parametres 
	!===================================================
	open(unit=11,file='data',status='old')
	read(11,*) Nx,Ny
	read(11,*) Lx,Ly
	read(11,*) D
	close(11)

	f1=2.d0*((x-x**2)+(y-y**2)) 
	!f1=sin(x)+cos(y)
    !f1=exp(-1*(x-0.5*Lx)**2)*exp(-1*(y-0.5*Ly)**2)*cos(0.5*pi*t)!sin(x)+cos(y)!2*((x-x**2)+(2*y-y**2)) 
  end function f1
!========================================================================
!fonction g
!========================================================================
function g(x,y)
    double precision :: x, y, g
    g=0.d0
    !g=sin(x)+cos(y)
  end function g
!=============================================
!fonction h
!=============================
 function h(x,y)
    double precision :: x, y, h
    h=0.0
    !h=1.d0
    !h=sin(x)+cos(y)
  end function h
!================================================
!subroutine passage permettant de passer
!de la num�rotation locale
!� la num�rotation globale
!==============================================
  subroutine passage(k,i,j,Nx)
    implicit none

    integer,intent(in):: k, Nx
	integer,intent(out):: i, j
      
    i= 1+int((k-1)/Nx)
    j= 1+mod(k-1,Nx)
  end subroutine passage

!========================================================
!fonction prodMV qui r�alise le produit matrice vecteur
!de A et de X
!=============================================
  
  function prodMV(A,X,n,Nx)
	double precision,dimension(1:5,1:n) :: A
	double precision,dimension(:), allocatable :: prodMV
	double precision,dimension(:), allocatable :: Xp
	double precision,dimension(1:n) :: X
	integer :: n, i, j, Nx

	allocate(prodMV(1:n))
  	allocate(Xp(1-Nx:n+Nx))
	Xp(1:n)=X
	
	prodMV=0

	do i=1,n

		prodMV(i)=A(1,i)*Xp(i-Nx)+A(2,i)*Xp(i-1)+A(3,i)*Xp(i)+A(4,i)*Xp(i+1)+A(5,i)*Xp(i+Nx)

	end do
	
 end function prodMV
!===============================================================
!fonction prodMV_para qui r�alise le produit matrice vecteur
!de A et de X dans le cadre du parallelisme
!=============================================
  
function prodMV_para(A,X,n,Nx,i1,ie)
	double precision,dimension(1:5,i1:ie) :: A
	double precision,dimension(:), allocatable :: prodMV_para
	double precision,dimension(:), allocatable :: Xp
	double precision,dimension(1:n) :: X
	integer :: n, i, j, Nx,i1, ie

	allocate(prodMV_para(i1:ie))
  	allocate(Xp(1-Nx:n+Nx))
	Xp(1:n)=X

	do i=i1,ie
			
			prodMV_para(i)=A(1,i)*Xp(i-Nx)+A(2,i)*Xp(i-1)+A(3,i)*Xp(i)+A(4,i)*Xp(i+1)+A(5,i)*Xp(i+Nx)

	end do
	
 end function prodMV_para
!=============================================================== 
!subroutine charge qui partage equitablement n en Np proceseurs
!=======================================================
 subroutine charge(Me,n,Np,i1,ie)
  implicit none

  integer,intent(in):: n,Np,Me
  integer,intent(out)::i1,ie
  integer::i

  if (Me<mod(n,Np)) then
  	i1=(1+(n/Np))*Me+1
    ie=(n/Np)+i1
      
  else
     
    i1=(n/Np)*Me+mod(n,Np)+1
    ie=i1+(n/Np)-1
    
  end if
  end subroutine charge
!=========================================================
end module mod_fonctions_instationnaire
  
  
  
  
  

