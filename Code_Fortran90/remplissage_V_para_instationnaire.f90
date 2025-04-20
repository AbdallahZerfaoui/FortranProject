program remplissage_V
  use mod_fonctions_instationnaire
  use cg_solver_parallel	
  use mod_remplissage_instationnaire
  !use mpi
  !use Grad
  
  implicit none
  include "mpif.h"

  integer :: Nx,Ny, nt=10,n, i, j, k, l, p, compt=0, Np, i1, ie,i0, Me, statinfo, j1, je, j0, r! k coordonnée globale 
  integer,dimension(MPI_STATUS_SIZE)::status
  double precision :: Lx, Ly, D, dx, dy, dt=0.d0, Epsi=1E-4, alpha=1
  double precision :: alpha_me_num,alpha_me_deno,alpha_num=0,alpha_deno=0,beta=1,&
  beta_me_num,beta_me_deno,beta_num=0,beta_deno=0, val_trans=0, norm_Gr_loc, norm_Gr=1.d0,&
  t1_b, t2_b, t1_e, t2_e
  double precision,dimension(:,:), allocatable :: A, Id, M
  double precision,dimension(:), allocatable :: U, Uk, F, Gr, Grk, DIR, V
  character*25 :: ch

!===================================================
!Recuperation des parametres 
!===================================================
  open(unit=11,file='data',status='old')
  read(11,*) Nx,Ny
  read(11,*) Lx,Ly
  read(11,*) D
  close(11)

  !call CPU_TIME(t1_b)

  t1_b=MPI_WTIME()

  dx=Lx/(Nx+1)
  dy=Ly/(Ny+1)
  
  n=Nx*Ny
!====================================================
	
  !==============================
  ! 		DEBUT   	!
  !============================== 

  CALL MPI_INIT(statinfo)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD,Me,statinfo)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD,Np,statinfo)
  
  !==============================
  ! PARTAGE DES CHARGES       	!
  !==============================  
  call charge(Me, n, Np, i1, ie)
  !==============================
  !ALLOCATION
  !==============================
  
  
  allocate(A(1:5,i1:ie))
  allocate(M(i1:ie,0:nt-1))
  allocate(Id(1:5,i1:ie))
  allocate(U(1:n))
  allocate(F(i1:ie))

  !allocate(Gr(i1:ie))
  !allocate(Grk(i1:ie))
  !allocate(DIR(1:n))
  allocate(Uk(1:n))
  !allocate(V(i1:ie))
  
  A=0
  U=0
  Uk=0
  F=0

  
  Id=0.d0
  M(:,0)=Uk(i1:ie)
  Id(3,:)=1
  do l=i1, ie
	Id(3,l)=1.d0
  end do
  
  !==============================
  ! Remplissage de la matrice A !
  !==============================
do k=2,n-1    ! on traite le cas i=1 et i=n comme cas particuliers 
	if (k>=i1 .and. k<=ie) then
		!print*, Me, 'Je suis laF'
		A(3,k)=-2*(dx**2+dy**2)
     	call passage(k, i, j, Nx)
     		if (j-1 >=1) then
        		A(2,k)=dy**2
     		else
		
        		!F(k)=F(k)+h(0.d0,i*dy)/(dx**2)
     		end if
     
     		if (j+1<=Nx) then
        		A(4,k)=dy**2
     		else
        
			    !F(k)=F(k)+h(Lx,i*dy)/(dx**2)
     		end if
	end if 
	
  ! jusque la on a rempli les cases a droite et a gauche
  
end do

  if (Me==0) then
  	A(3,1)=-2*(dx**2+dy**2)
  	A(4,1)=dy**2
  	A(5,1)=dx**2
  end if
  if (Me==Np-1) then
  	A(3,n)=-2*(dx**2+dy**2)
  	A(2,n)=dy**2
  	A(1,n)=dx**2
  end if

	
do k=1,n-Nx
	if (k>=i1 .and. k<=ie) then
     !call passage(k, i, j, Nx)
		A(5,k)=dx**2
	end if
end do
  
   
do k=1+Nx,n
	if (k>=i1 .and. k<=ie) then
     !call passage(k, i, j, Nx)
		A(1,k)=dx**2
	end if
end do

!call remplissage_F(F,n,Nx,Ny,Lx,Ly,Np,Me,i1,ie)
  A=-1*D*A/((dx*dy)**2)
  
!call remplissage_F(F,n,Nx,Ny,Lx,Ly,Np,Me,i1,ie)
!call CPU_TIME(t1_e)
t1_e=MPI_WTIME()	

 

  !================================================
  !sauvgarde de la matrice dans un fichier séparé !
  !================================================
  ch='Mat'//char(Me+48)
 ! print*, ch
  open(unit=69+Me,file=ch,status='unknown')
  do l=1, 5
     	write(69+Me,*) A(l,:)
  end do
  close(69+Me)
 
!call CPU_TIME(t2_b)
t2_b=MPI_WTIME()
do k=0, nt-1	!BOUCLE DU TEMPS
  !==============================
  ! Remplissage du vecteur F 	!
  !==============================
  
call remplissage_F(F,n,Nx,Ny,Lx,Ly,k,dt,Np,Me,i1,ie)

  !==============================
  ! Sauvegarde du vecteur F     !
  !==============================
 	ch='VectF'//char(Me+48)
	print*, ch
	open(unit=691+Me,file=ch,status='unknown')
	do l=i1,ie
		call passage(l,i,j,Nx)
		write(691+Me,*) j*dx,i*dy,F(l)
	end do
	close(691+Me) 
  !===============================================================
  !RESOLUTION DU SYSTEME AU=F PAR LA METHODE DU GRADIENT CONJUGE !
  !===============================================================

CALL Grad_conjuge_para(Id+dt*A,U,Uk+dt*F,n,Nx,Np,Me,i1,ie)
Uk=U
M(:,k)=U(i1:ie)
	
print*,'k est de ',k

end do !FIN DE LA BOUCLE


t2_e=MPI_WTIME()
print*, 'je suis le' ,Me, 1000.d0*(t1_e + t2_e - t1_b - t2_b)
print*, 'je suis le' ,Me, 1000.d0*(t1_e - t1_b)
print*, 'je suis le' ,Me, 1000.d0*(t2_e - t2_b)
!==============================
!SAUVGARDE du vecteur U local !
!==============================

	ch='VectU'//char(Me+48)
	print*, ch
	open(unit=690+Me,file=ch,status='unknown')
	do l=1, n
		call passage(l,i,j,Nx)
		write(690+Me,*) j*dx,i*dy,U(l)
	end do
	close(690+Me)  
!==============================
! ENVOI du vecteur U          !
!==============================

!call MPI_BARRIER(MPI_COMM_WORLD,statinfo)

print*, Me, 'good_avant'

	if (Me>0) then

		Call MPI_SEND(U(i1:ie), ie-i1+1,MPI_DOUBLE_PRECISION,0,101,MPI_COMM_WORLD,statinfo)	
	else 
		do l=1,Np-1
			call charge(l, n, Np, j1, je)
			Call MPI_RECV(U(j1:je), je-j1+1,MPI_DOUBLE_PRECISION,l,101,MPI_COMM_WORLD,status,statinfo)
			
		end do
		
	end if
!end if 
  !===============================
if (Me==0) then
  !==============================
  ! Sauvegarde du vecteur U     !
  !==============================
  open(unit=179,file='VecteurU_b',status='unknown')
  open(unit=177,file='VecteurU',status='unknown')
  do l=1, n
     write(177,*) U(l)
     call passage (l,i0,j0,Nx)
     write(179,*) j0*dx, i0*dy, U(l) 
  end do
end if
  close(177)
  close(179)
  close(175)

 
!=========================================
!%%%%%%%%%%%% THE END %%%%%%%%%%%%%%%%%%%!      
!=========================================
    CALL MPI_FINALIZE(statinfo)
  !=========================================
  !=========================================

  deallocate(A)
  deallocate(U)
  deallocate(F)
  !deallocate(Gr)
  !deallocate(Grk)
  !deallocate(DIR)
  !deallocate(Uk)
  !deallocate(V)
  
end program remplissage_V
