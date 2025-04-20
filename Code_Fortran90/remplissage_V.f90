 program remplissage_V
  use mod_fonctions_instationnaire
  use cg_solver_sequential
  
  implicit none

  integer :: Nx,Ny, n, i, j, k, l, compt=0, i0, j0   ! k coordonnée globale 
  integer :: nt=13

  double precision :: Lx, Ly, D, dx, dy, dt=1, Epsi=1E-4, alpha=1, p
  double precision :: t1_b, t2_b, t1_e, t2_e 
  double precision,dimension(:,:), allocatable :: A,Id, M
  double precision,dimension(:), allocatable :: U, Uk, F, Gr, Grk, DIR, V
  
!===================================================
!Recuperation des parametres 
!===================================================
  open(unit=11,file='data',status='old')
  read(11,*) Nx,Ny
  read(11,*) Lx,Ly
  read(11,*) D
  close(11)
  
  call CPU_TIME(t1_b)
  dx=Lx/(Nx+1)
  dy=Ly/(Ny+1)
!====================================================
  
  n=Nx*Ny
  allocate(A(1:5,1:n))
  allocate(M(1:n,0:nt-1))
  allocate(Id(1:5,1:n))
  allocate(U(1:n))
  allocate(F(1:n))
  allocate(Gr(1:n))
  allocate(Grk(1:n))
  allocate(DIR(1:n))
  allocate(Uk(1:n))
  allocate(V(1:n))
  print*, dx, dy
  
  A=0
  U=0
  Uk=1
  F=0
  Gr=0
  DIR=0
  !DIRk=0
  V=0
  
  Id=0.d0
  M(:,0)=Uk
  Id(3,:)=1
  do l=1, n
	Id(3,l)=1.d0
  end do
  !==============================
  ! Remplissage de la matrice A !
  !==============================
  do k=2,n-1    ! on traite le cas i=1 et i=n comme cas particuliers 
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
  ! jusque la on a rempli les cases a droite et a gauche
  end do
  
  A(3,1)=-2*(dx**2+dy**2)
  A(4,1)=dy**2
  A(5,1)=dx**2
  A(3,n)=-2*(dx**2+dy**2)
  A(2,n)=dy**2
  A(1,n)=dx**2

  !A(1,Nx+1:n)=1
  !A(2,2:n)=1
  !A(3,1:n)=-4
  !A(4,1:n-1)=1
  !A(5,1:Nx)=1
	
  do k=1,n-Nx
     !call passage(k, i, j, Nx)
     A(5,k)=dx**2
  end do
  
    
  do k=1+Nx,n
     !call passage(k, i, j, Nx)
     A(1,k)=dx**2
  end do
  
  A=-1*D*A/((dx*dy)**2)
  
  call CPU_TIME(t1_e)	

do k=0, nt-1	!BOUCLE DU TEMPS
  !==============================
  ! Remplissage du vecteur F 	!
  !==============================
  
  do l=1,n
	call passage(l,i,j,Nx) 
	F(l)=f1(j*dx,i*dy,(k+1)*dt)
  end do
  !print*, F(1)
  do l=1, Nx				! pour prendre les bords haut et bas en consideration
	F(l)=F(l)+g(l*dx,0.d0)/(dy**2)
  end do
 
  do l=n-Nx+1, n
	F(l)=F(l)+g((l-n+Nx)*dx,Ly)/(dy**2)
	!print*, 'c g',g((l-n+Nx)*dx,Ly)
  end do	
 do l=Nx, n, Nx				! pour prendre les bords haut et bas en consideration
	call passage(l,i0,j0,Nx)
	
	 F(l)=F(l)+h(Lx,i0*dy)/(dx**2)
 end do
  	
  
  do l=1, n-Nx+1, Nx				! pour prendre les bords haut et bas en consideration
	 call passage(l,i0,j0,Nx)
	
	 F(l)=F(l)+h(0.d0,i0*dy)/(dx**2)
  end do
 ! LE CAS PARTICULIER 1 et n qui vont intervenir la fonction h
  !F(1)=F(1)+h(0.d0,dy)/(dx**2)
 
  !F(n)=F(n)+h(0.d0,n*dy)/(dx**2)

  !==================================
  ! GRADIENT CONJUGEE
  !==================================
	!M(:,k)=Uk
	
	call Grad_conjuge(Id+dt*A,U,Uk+dt*F,n,Nx)
	Uk=U
	M(:,k)=U
	compt=compt+1
	print*,k

end do  ! FIN BOUCLE TEMPS
  !==============================
  ! Sauvegarde du vecteur F		!
  !==============================
  
  open(unit=169,file='VecteurF',status='unknown')
  do l=1, n
     write(169,*) F(l)
  end do
  close(169)

  !================================================
  !sauvgarde de la matrice dans un fichier séparé !
  !================================================
  open(unit=69,file='Matrice_Quadri',status='unknown')
  do l=1, 5
     write(69,*) A(l,:)
  end do
  close(69)

  call CPU_TIME(t2_b)
  
  call CPU_TIME(t2_e)

  print*, 'je suis le', 1000*(t1_e + t2_e - t1_b - t2_b)
  !==============================
  ! Sauvegarde du vecteur U 	!
  !==============================
 
  open(unit=179,file='VecteurU_b',status='unknown')
  open(unit=177,file='VecteurU',status='unknown')
  do l=1, n
     write(177,*) U(l)
     call passage (l,i0,j0,Nx)
     write(179,*) j0*dx, i0*dy, U(l)
  end do
  close(177)
  close(179)
 
  
  !=========================================
  ! SAUVGARDE DE M
  !========================================
  open(unit=1779,file='VecteurM',status='unknown')
  do l=1, n
     
     call passage (l,i0,j0,Nx)
     write(1779,*) j0*dx, i0*dy, M(l,:)
  end do
  
  close(1779)
  
  
  deallocate(A)
  deallocate(Id)
  deallocate(U)
  deallocate(F)
  deallocate(Gr)
  deallocate(Grk)
  deallocate(DIR)
  deallocate(Uk)
  deallocate(V)
  
end program remplissage_V
