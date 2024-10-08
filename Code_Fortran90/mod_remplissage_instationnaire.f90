module mod_remplissage_instationnaire 
	use mod_fonctions_instationnaire
	implicit none 
  
contains
	subroutine remplissage_F(F,n,Nx,Ny,Lx,Ly,k,dt,Np,Me,i1,ie)
    implicit none
    
    integer, intent(in) :: n,Nx,Ny,Me, Np
	integer :: i, j, k, l, p, i1, ie,i0, j1, je, j0, r ! k coordonnée globale 
	!double precision,dimension(1:5,1:n), intent(in) :: A
	double precision,dimension(i1:ie), intent(inout) :: F
	!double precision,dimension(1:n), intent(in) :: F
	!double precision,dimension(1:n)::Gr,DIR,V,Uk,Grk
	double precision::alpha,epsi=1E-4
	double precision :: dx,dy,dt,Lx,Ly
  	double precision :: alpha_me_num,alpha_me_deno,alpha_num=0,alpha_deno=0,beta=1,&
  	beta_me_num,beta_me_deno,beta_num=0,beta_deno=0, norm_Gr_loc, norm_Gr=1.d0
	
	dx=Lx/(Nx+1)
  	dy=Ly/(Ny+1)
	!==============================
  	! Remplissage du vecteur F 	!
  	!==============================
  
do l=1,n
	if (l>=i1 .and. l<=ie) then 
		call passage(l,i,j,Nx) 
		F(l)=f1(j*dx,i*dy,(k+1)*dt)
	end if
end do
  !print*, F(1)
do l=1, Nx				! pour prendre les bords haut et bas en consideration
	if (l>=i1 .and. l<=ie) then 
		F(l)=F(l)+g(l*dx,0.d0)/(dy**2)
	end if
end do
 
do l=n-Nx+1, n
	if (l>=i1 .and. l<=ie) then 
		F(l)=F(l)+g((l-n+Nx)*dx,Ly)/(dy**2)
	!print*, 'c g',g((l-n+Nx)*dx,Ly)
	end if
end do	
do l=Nx, n, Nx				! pour prendre les bords haut et bas en consideration
	if (l>=i1 .and. l<=ie) then 
		call passage(l,i0,j0,Nx)
	
		F(l)=F(l)+h(Lx,i0*dy)/(dx**2)
	end if
end do
  	
  
do l=1, n-Nx+1, Nx				! pour prendre les bords haut et bas en consideration
	if (l>=i1 .and. l<=ie) then 
		call passage(l,i0,j0,Nx)
	
		F(l)=F(l)+h(0.d0,i0*dy)/(dx**2)
	end if
end do	
 
 ! LE CAS PARTICULIER 1 et n qui vont intervenir la fonction h
  !if (Me==0) then 
  	!F(1)=F(1)+h(0.d0,dy)/(dx**2)
  !end if
  !if (Me==Np-1) then
  	!F(n)=F(n)+h(0.d0,n*dy)/(dx**2)
  !end if

	
	end subroutine remplissage_F
end module mod_remplissage_instationnaire
