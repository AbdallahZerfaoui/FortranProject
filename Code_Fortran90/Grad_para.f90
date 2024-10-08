module Grad_para
	use mod_fonctions_instationnaire
	implicit none 
  
contains
	subroutine Grad_conjuge_para(A,U,F,n,Nx,Np,Me,i1,ie)
    implicit none
    include "mpif.h"

!============================================
!Déclaration des variables
!============================================
    integer, intent(in) :: n,Nx
	double precision,dimension(1:5,i1:ie), intent(in) :: A
	double precision,dimension(1:n), intent(inout) :: U
	double precision,dimension(i1:ie), intent(in) :: F
	double precision,dimension(i1:ie)::Gr,V,Grk
	double precision,dimension(1:n):: DIR
	double precision::alpha,p,Epsi=1E-4
	double precision:: t1_b,t2_b,t3_b,t1_e,t2_e,t3_e
  	double precision :: alpha_me_num,alpha_me_deno,alpha_num=0,alpha_deno=0,beta=1,&
  	beta_me_num,beta_me_deno,beta_num=0,beta_deno=0, norm_Gr_loc, norm_Gr=1.d0
	integer::compt,l,Me,i1,ie, statinfo, Np
	
!==============================
! Initialisation de l'algorithme
!==============================
DIR=0.d0
Gr(i1:ie)=prodMV_para(A,U,n,Nx,i1,ie)-1.d0*F(i1:ie) 
DIR(i1:ie)= -Gr(i1:ie)

!==============================
!Début de l'alogorithme
!==============================
!Condition d'arret de l'algorithme
do while (norm_Gr>Epsi)
	
!==============================
!Communication du vecteur de direction
!==============================
!Communication des blocs de Nx suivants la partie concerné 
!sauf en pour Me=Np-1	
if (Me<Np-1) then !suivant
		
		call MPI_SENDRECV(DIR(ie-Nx+1:ie), Nx, MPI_DOUBLE_PRECISION, Me+1, 101, DIR(ie+1:ie+Nx), &
		Nx, MPI_DOUBLE_PRECISION, Me+1, 101, MPI_COMM_WORLD,MPI_STATUS_IGNORE, statinfo)
		
	end if
	
!Communication des blocs de Nx précédents la partie concerné 
!sauf en pour Me=0	
	if (Me>0) then 	
		call MPI_SENDRECV(DIR(i1:i1+Nx-1), Nx, MPI_DOUBLE_PRECISION, Me-1, 101, DIR(i1-Nx:i1-1), &
		Nx, MPI_DOUBLE_PRECISION, Me-1, 101, MPI_COMM_WORLD,MPI_STATUS_IGNORE,statinfo)
		
	end if
	

!Produit matrice vecteurentre A et le vecteur DIR	
V=prodMV_para(A,DIR,n,Nx,i1,ie)
	
!==============================
! Calcul de alpha
!==============================
!Calcul de la partie numérateur	
alpha_me_num=dot_product(Gr(i1:ie),Gr(i1:ie))
!Calcul de partie  denominateur	
alpha_me_deno=dot_product(DIR(i1:ie),V(i1:ie)) 
	
!Envoi et sommation des parties numerateurs de chaque processeur	
call MPI_ALLREDUCE(alpha_me_num,alpha_num,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,statinfo)
	
!Envoi et sommation des parties dénominateurs de chaque processeur
    call MPI_ALLREDUCE(alpha_me_deno,alpha_deno,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,statinfo)
	
alpha=alpha_num/alpha_deno

!%==========================
!% MISE A JOUR DE U %
!%==========================
	
U(i1:ie)=U(i1:ie)+alpha*DIR(i1:ie)

!%==========================
!% MISE A JOUR DU GRADIENT %
!%==========================
	Grk(i1:ie)=Gr(i1:ie)
	Gr(i1:ie)=Gr(i1:ie)+alpha*V(i1:ie)
    !%==================================
    !% CALCUL DE LA NOUVELLE DIRECTION %
    !%==================================
!==============================
! Calcul de beta
!==============================
!Calcul de la partie numérateur
beta_me_num = dot_product(Gr(i1:ie),Gr(i1:ie))
!Calcul de la partie dénominateur
beta_me_deno = dot_product(Grk(i1:ie),Grk(i1:ie))
	
!Envoie et sommation des parties numerateurs de chaque processeur
	call MPI_ALLREDUCE(beta_me_num,beta_num,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,statinfo)
	
!Envoie et sommation des parties dénominateurs de chaque processeur
	call MPI_ALLREDUCE(beta_me_deno,beta_deno,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,statinfo)
	
	beta=beta_num/beta_deno

!mise à jour de la direction
	DIR(i1:ie)=-1.d0*Gr(i1:ie)+beta*DIR(i1:ie)
    
!%==================================
    !% CALCUL DE LA NORME DU GRADIENT %
    !%==================================
	norm_Gr_loc=sqrt(dot_product(Gr(i1:ie),Gr(i1:ie)))
	
!Envoi et sommation des norms de gradient de chaque processeur
	call MPI_ALLREDUCE(norm_Gr_loc,Norm_Gr,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,statinfo)
	
end do
	

	end subroutine Grad_conjuge_para
end module Grad_para

