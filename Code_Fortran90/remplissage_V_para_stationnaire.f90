program remplissage_V
  use mod_fonctions_instationnaire
  use cg_solver_parallel	
  use mod_remplissage
  implicit none
  include "mpif.h"

!===================================================
!Déclaration des variables 
!===================================================
  integer :: Nx,Ny, n, i, j, k, l, p, compt=0, Np, i1, ie,i0, Me, statinfo, j1, je, j0, r 
  integer,dimension(MPI_STATUS_SIZE)::status
  double precision :: Lx, Ly, D, dx, dy, dt=1, Epsi=1E-4, alpha=1
  double precision :: alpha_me_num,alpha_me_deno,alpha_num=0,alpha_deno=0,beta=1,&
  beta_me_num,beta_me_deno,beta_num=0,beta_deno=0, val_trans=0, norm_Gr_loc, norm_Gr=1.d0,&
  t1_b, t2_b, t1_e, t2_e 
  double precision,dimension(:,:), allocatable :: A
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

!===================================================
! Declenchement du chronometre
! pour le remplissage
!===================================================
  t1_b=MPI_WTIME()
!===================================================
!Calcul de dx et dy 
!===================================================
  dx=Lx/(Nx+1)
  dy=Ly/(Ny+1)
  
  n=Nx*Ny


  !====================================
  ! 		DEBUT  DU PARALLELISME 	!
  !==================================== 

  CALL MPI_INIT(statinfo)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD,Me,statinfo)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD,Np,statinfo)
  
  !==============================
  ! PARTAGE DES CHARGES       	!
  !==============================  
  call charge(Me, n, Np, i1, ie)
  !==========================================
  !ALLOCATION DE LA MATRICE ET DES VECTEURS
  !============================================
  
  
  allocate(A(1:5,i1:ie))
  allocate(U(1:n))
  allocate(F(i1:ie))
  
  A=0
  U=0
  F=0 
  
  
    !==============================
  ! Remplissage de la matrice A !
  !==============================
do k=2,n-1    ! on traite le cas i=1 et i=n comme cas particuliers 
	if (k>=i1 .and. k<=ie) then
           !remplissage de la diagonale i,i
		A(3,k)=-2*(dx**2+dy**2)
     	     call passage(k, i, j, Nx)
           !remplissage de la diagonal i,i-1
     		if (j-1 >=1) then
        		A(2,k)=dy**2
     		end if
           !remplissage de la diagonal i,i+1
     		if (j+1<=Nx) then
        		A(4,k)=dy**2
     		end if
	end if 
end do

!remplissage de la 1ere ligne de la « vraie » matrice
!par le processeur 0
  if (Me==0) then
  	A(3,1)=-2*(dx**2+dy**2)
  	A(4,1)=dy**2
  	A(5,1)=dx**2
  end if

!remplissage de la derniere ligne de la « vraie » matrice
!par le processeur Np-1
  if (Me==Np-1) then
  	A(3,n)=-2*(dx**2+dy**2)
  	A(2,n)=dy**2
  	A(1,n)=dx**2
  end if

!Remplissage de la diagonale superieure	
do k=1,n-Nx
	if (k>=i1 .and. k<=ie) then
		A(5,k)=dx**2
	end if
end do
  
   
!Remplissage de la diagonale inferieure
do k=1+Nx,n
	if (k>=i1 .and. k<=ie) then
		A(1,k)=dx**2
	end if
end do

!Mise à jour de A
 A=-1*D*A/((dx*dy)**2)
  
!===================================================
! Fin du chronometre du remplissage
!===================================================
t1_e=MPI_WTIME()	
!==============================
  ! Remplissage du vecteur F 
  !par la subroutine remplissage_F
  !==============================
call remplissage_F(F,n,Nx,Ny,Lx,Ly,Np,Me,i1,ie)
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

  !================================================
  !sauvegarde de la matrice dans un fichier séparé !
  !================================================
  ch='Mat'//char(Me+48)
   open(unit=69+Me,file=ch,status='unknown')
  do l=1, 5
     	write(69+Me,*) A(l,:)
  end do
  close(69+Me)
 
!===================================================
! Declenchement du chronometre pour le gradient
!===================================================
t2_b=MPI_WTIME()
  !===============================================================
  !RESOLUTION DU SYSTEME AU=F PAR LA METHODE DU GRADIENT CONJUGE !
  !===============================================================
CALL Grad_conjuge_para(A,U,F,n,Nx,Np,Me,i1,ie)

!===================================================
! Fin du chronometre pour le gradient
!===================================================
t2_e=MPI_WTIME()

!===================================================
! Calcul des différents temps d’execution
!===================================================
!Dans l’ordre
!temps de calcul pour le programme entier
!temps de calcul pour le remplissage
!temps de caclcul pour le gradient
print*, 'je suis le' ,Me, 1000.d0*(t1_e + t2_e - t1_b - t2_b)
print*, 'je suis le' ,Me, 1000.d0*(t1_e - t1_b)
print*, 'je suis le' ,Me, 1000.d0*(t2_e - t2_b)
!==============================
!SAUVGARDE du vecteur U local !
!==============================

	ch='VectU'//char(Me+48)
	open(unit=690+Me,file=ch,status='unknown')
	do l=1, n
		call passage(l,i,j,Nx)
		write(690+Me,*) j*dx,i*dy,U(l)
	end do
	close(690+Me)  
!==============================
! ENVOI du vecteur U          !
!==============================


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
!%%%%%%%%%%%% Fin du parallelisme%%%%%%%%%%%%%%%%%%%!      
!=========================================
    CALL MPI_FINALIZE(statinfo)
  !============================================
  !DESALLOCATION DE LA MATRICE ET DES VECTEURS
!==============================================
   deallocate(A)
  deallocate(U)
  deallocate(F)
  
end program remplissage_V

