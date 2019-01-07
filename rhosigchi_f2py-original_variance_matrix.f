
      SUBROUTINE rhosigchi(Fg_in,calib_in,Eg_min,Ex_min,Ex_max,
     &                     Rho_fin,Sig_fin,calib_out)
C Read/write stuff (mama)
CJEM      COMMON/Sp1Dim/rSPEC(2,0:8191),MAXCH
CJEM      COMMON/Sp2Dim/rMAT(2,0:4095,0:511),APP(512),XDIM,YDIM
CJEM      COMMON/Sp2Dim/rMAT(2,0:4095,0:511),APP(512),XDIM,YDIM
      INTEGER XDIM,YDIM,MAXCH
CJEM      COMMON/State/Istatus,ITYPE,IDEST,cal(2,2,2,3),Idim(2,2,2),fname(2,2),comm(2,2)
      CHARACTER fname*8,comm*60,comment*60
      CHARACTER outfile*20,APP*4
      COMMON/iter/jmin,jmax,igmin,nit,igmax(0:511),iu0

C Stuff for the rhosig iteration
      REAL Rho(0:100,0:511),Rhov(0:100,0:511)
      REAL Sig(0:100,0:511),Sigv(0:100,0:511)
      REAL SumFg(0:511), SumFgv(0:511)
      REAL Fg(0:511,0:511),FgTeo(0:511,0:511),FgN(0:511,0:511)
      REAL Fgv(0:511,0:511),sFg(0:511,0:511),sFgN(0:511,0:511)
CJEM  Added some variables and f2py magic lines:
      REAL calib_in(4)
      REAL Fg_in(0:511,0:511)
CJEM 2019 cannot see that this is needed? REAL sSum, sFi(0:511), sFf(0:511)
      REAL calib_out(2)
CJEM  calib_in contains calibration coefficients in the order (aEg0, aEg1,
CJEM  aEx0, aEx1), where Ei = aEi1*channel + aEi0
CJEM  calib_out contains common calibration coefficients in the order (a0, a1)
Cf2py REAL intent(in) :: Fg_in, calib_in
      REAL Rho_fin(0:100), Sig_fin(0:100)
Cf2py REAL intent(out) :: Rho_fin, Sig_fin, calib_out
      REAL sRho(0:511),sSig(0:511)
      REAL rRho(0:511),rSig(0:511),Sum,Sum2
      REAL Chi(0:100),a1,a0
      REAL Eg_min,Ex_min,Ex_max,Emr
Cf2py REAL intent(in) :: Eg_min, Ex_min, Ex_max
      REAL Ex,dE_p,dE_g1MeV,dE_g8MeV,dE_g
      REAL egmax(0:511)
CJEM      INTEGER igmax(0:511)
      INTEGER time,nunc,degrees,deg,jmax,ix,ig,iu,iemr,step,in0
      INTEGER ism,istart,istop,ll,i,j
      REAL*8 xfit(1:512),yfit(1:512),y0,y1,y2,x0,x1,x2,x3,x4,d0,d1,d2,d3 !REAL->REAL*8
      REAL*8 tmp(0:511),coef(3)                                          !Corrected 11 feb 2016
      DIMENSION Fi(0:511),Ff(0:511) 
      WRITE(6,*)' ________________________________________'
      WRITE(6,*)'|                                        |'
      WRITE(6,*)'|         R H O S I G C H I  1.5.3       |'
      WRITE(6,*)'|                                        |'
      WRITE(6,*)'|  Program to calculate level density    |'
      WRITE(6,*)'| Rho, and gamma strength function Sig   |'
      WRITE(6,*)'| from first-generation spectra, using   |'
      WRITE(6,*)'|   FgNorm(Ex,Eg)=Rho(Ex-Eg)*Sig(Eg)     |'
      WRITE(6,*)'|                                        |'
      WRITE(6,*)'|      Oslo Cyclotron Laboratory         |'
      WRITE(6,*)'|                                        |'
      WRITE(6,*)'|        Created: 13/08 - 1999           |'
      WRITE(6,*)'|           Andreas Schiller             |'
      WRITE(6,*)'|  Lisbeth Bergholt, Magne Guttormsen    |'
      WRITE(6,*)'|                                        |'
      WRITE(6,*)'|        Improved: 01/05 - 2003          |'
      WRITE(6,*)'|           Alexander Voinov             |'
      WRITE(6,*)'|  Magne Guttormsen, Andreas Schiller    |'
      WRITE(6,*)'|        Dubna, Oslo, Livermore          |'
      WRITE(6,*)'|  Modified 28 Mar 2014: Ex < 0 MeV      |'
      WRITE(6,*)'|  Modified 13 Feb 2015: One ch shift,   |'
      WRITE(6,*)'|  ch ig < 0, cut igmax                  |'
      WRITE(6,*)'|  Modified 31 Aug 2015: No normalization|'
      WRITE(6,*)'|  Modified 15 Dec 2015: input > 10**-06 |'
      WRITE(6,*)'|  Modified 11 Feb 2016: new Ydim, Xdim  |'
      WRITE(6,*)'|  REAL->REAL*8 precision, smooth 300 keV|'
      WRITE(6,*)'|________________________________________|'
C Things that should be done in the future:
C Fitting with error weighting
C Finding a better parametrization of the strength function
C Initializing parameter values
      nit=50                  ! Number of iterations
CJEM      Ex_min=4000.0           ! Default lower Ex for 1.gen. spec.
CJEM       Ex_max=8000.0           ! Default higher Ex for 1.gen. spec.
CJEM       Eg_min=1000             ! Default lowest gamma energy
      nunc=100                ! Number of extractions
      dE_p = 150.             ! Particle resolution
      dE_g1MeV = 60.          ! NaI resolution at 1 MeV
      dE_g8MeV = 300.         ! NaI resolution at 8 MeV
      eps      = 0.000001     ! Lowest counts in fg matrix

C Initializing some vectors and matrices
      DO i=0,511
        sRho(i) = 0.
        sSig(i) = 0.
        rRho(i) = 0.
        rSig(i) = 0.
        DO j=0,100
           Rho(j,i)  = 0.
           Rhov(j,i) = 0.
           Sig(j,i)  = 0.
           Sigv(j,i) = 0.
        ENDDO
      ENDDO

C Reading first-generation mama-matrix
C       IDEST=1
C       ITYPE=3
C       WRITE(6,*)'Please, answer 1 and the name of your input first-'
C       WRITE(6,*)'generation matrix in the two next questions... '
CJEM      CALL READFILE("firstgen-jem-20161011-Nexbins196-Nstat500.m")
CJEM   added to load fg variance (fgv) spectrum into the other rMAT channel:
CJEM      IDEST = 2
CJEM      CALL READFILE("fgvar-jem-20161011-Nexbins196-Nstat500.m")
CJEM      IDEST = 1
CJEM  done loading fgv    
CJEM      IF(XDIM.GT.512)XDIM=512
CJEM      IF(YDIM.GT.512)YDIM=512
CJEM  20161114: Replaced the two lines above with hard-coded values
      XDIM = 512
      YDIM = 512
CTODO 20161104
CTODO replace cal() by calibrations input through rhosigchi() subroutine
CTODO 
CJEM      bx = calib(2)
CJEM      by = calib(4)
CJEM  Added by JEM: Calibration coefficients for gamma and ex axes
CJEM  read from input argument. Specification: Ei = aEi1*channel + aEi0
      aEg0 = calib_in(1)
      aEg1 = calib_in(2)
      aEx0 = calib_in(3)
      aEx1 = calib_in(4)

      write(6,*)'Calibration coefficients: ',aEg0, aEg1, aEx0, aEx1

CJEM      bx=cal(1,IDEST,1,1)+cal(1,IDEST,1,2)+cal(1,IDEST,1,3)
CJEM      by=cal(1,IDEST,2,1)+cal(1,IDEST,2,2)+cal(1,IDEST,2,3)
CJEM      IF(bx+by.EQ.2.)THEN
CJEM        aEg0=11.               !Defaults in case of no calibration
CJEM        aEg1=20.
CJEM        aEx0=9660.
CJEM        aEx1=-120.
CJEM      ELSE
CJEM        aEg0=cal(1,IDEST,1,1)  !Picks the spectrum calibration
CJEM        aEg1=cal(1,IDEST,1,2)
CJEM        aEx0=cal(1,IDEST,2,1)
CJEM        aEx1=cal(1,IDEST,2,2)
CJEM      ENDIF
      a1=ABS(aEx1)
      multiple=INT((120./a1)+0.5)
      IF(a1.GT.150.)multiple=1
      a1=FLOAT(multiple)*a1
      write(6,*)'multiple = ', multiple, 'a1 = ', a1
C An energy that was in the middle of a channel, shall still be in the
C middle of a channel after change of calibration
      Eold=aEx0+aEx1*0                                  !Choosing old channel 0
      a0=(Eold/a1-INT(Eold/a1))*a1

C Now calculating max energy for NaI to cover resolution. Max is 800 keV
      a0  = a0 - INT((800./a1) + 0.5)*a1

CJEM addition: return a0 and a1 as calib_out
      calib_out = (/ a0, a1 /)


      iu0 = INT((ABS(a0/a1) + 0.5))
      DO j=0, 511
        Ex = a0 + j*a1
        dE_g = dE_g1MeV + ((dE_g8MeV-dE_g1MeV)/(8000.-1000.))*(Ex-1000.)
        egmax(j) = SQRT(dE_g*dE_g + dE_p*dE_p)
CJEM  Added write for debug:
CJEM        write(6,*)'j = ',j,' egmax(j) = ',egmax(j)
        IF(egmax(j).GT.600.)egmax(j)=600.
        IF(egmax(j).LT.200.)egmax(j)=200.
        igmax(j) = j + INT((egmax(j)/a1)+0.5)
        IF(igmax(j).GT.511)igmax(j)=511
        egmax(j) = Ex + egmax(j)
      ENDDO

C Compressing (or stretching) along X and Y - axis
      DO i=0,511
         Fi(i)=0.
         Ff(i)=0.
CJEM          sFi(i)=0.
CJEM         sFf(i)=0.
      ENDDO

      DO j=0,YDIM-1
         DO i=0,XDIM-1
CJEM            If(rMAT(IDEST,i,j).LT.eps) rMAT(IDEST,i,j) = eps
CJEM  Changed rMAT to Fg_in (should sFg_in get same tratment?)
            If(Fg_in(i,j).LT.eps) Fg_in(i,j) = eps
CJEM  Write test:
C             if(Fg_in(i,j).GT.eps) write(6,*)'Fg_in(',i,j,')=',Fg_in(i,j)
         ENDDO
      ENDDO
      DO j=0,YDIM-1
         Sum=0.
CJEM         sSum=0.
         DO i=0,XDIM-1
CJEM            Fi(i)=rMAT(IDEST,i,j)                       ! Fi(i) and Ff(i) real type
CJEM  Changed rMAT to Fg_in
               Fi(i) = Fg_in(i,j)
            Sum=Sum+Fi(i)
CJEM            sSum=sSum+sFi(i)
         ENDDO
         IF(Sum.NE.0.)THEN
            CALL ELASTIC(Fi,Ff,aEg0,aEg1,a0,a1,512,512) ! Modifies spectrum to give  
            DO i=0,XDIM-1                               ! calibration a0 and a1
               Fg(i,j)=Ff(i)
               Fi(i)=0.
            ENDDO
         ENDIF
      ENDDO

      DO i=0,511
         Fi(i)=0.
         Ff(i)=0.
CJEM         sFi(i)=0.
CJEM         sFf(i)=0.
      ENDDO

      DO i=0,XDIM-1                                     ! Y-axis
         Sum=0.
CJEM         sSum=0.
         DO j=0,YDIM-1
            Fi(j)=Fg(i,j)
            Sum=Sum+Fi(j)
         ENDDO
         IF(Sum.NE.0.)THEN
            CALL ELASTIC(Fi,Ff,aEx0,aEx1,a0,a1,512,512)
            DO j=0,YDIM-1
               Fg(i,j)=Ff(j)
C                if(Fg(i,j).gt.0) write(6,*)'Fg(',i,j,')=',Fg(i,j)
               Fi(j)=0.
            ENDDO
         ENDIF
      ENDDO

C Replacing negative counts with 0 and finding dimension of Fg matrix
C      XDIM=INT(FLOAT(XDIM)*ABS(aEg1/a1)+0.5) + iu0
c      YDIM=INT(FLOAT(YDIM)*ABS(aEx1/a1)+0.5)
c      YDIM=INT((ABS((FLOAT(YDIM) * aEx1 + aEx0 - a0))/ABS(a1)) + 0.5)
c      XDIM=YDIM + iu0
      XDIM=INT((ABS((FLOAT(XDIM) * aEg1 + aEg0 -a0))/ABS(a1))+0.5)+ iu0
      YDIM=INT((ABS((FLOAT(YDIM) * aEx1 + aEx0 - a0))/ABS(a1)) + 0.5)
CJEM  20161114: Hard-coded XDIM and YDIM since I'm always sending in 512x512 stretched and ready
CJEM      XDIM = 512
CJEM      YDIM = 512
      write(6,*)'XDIM =',XDIM,' YDIM = ',YDIM

c           write(6,*)xdim,ydim
      imax=10
      DO j=0,YDIM
         DO i=0,XDIM
CJEM            if(Fg(i,j).gt.0)write(6,*)Fg(i,j)
            IF(Fg(i,j).GT.0..AND.i.GT.imax)imax=i
            IF(Fg(i,j).LE.0.)Fg(i,j)=0             !Delete negative numbers
         ENDDO
      ENDDO

      imax=MIN(imax,XDIM)
      Eg_limit = a0+a1*imax





c      write(6,*)'imax,XDIM, Eglimit',imax,XDIM, Eg_limit

CJEM       OPEN(23,FILE='input.rsg',STATUS='old',ERR=666)
CJEM       READ(23,*,END=666,ERR=666)Eg_min,Ex_min,Ex_max
CJEM       GO TO 777
CJEM  666  WRITE(6,*)'Warning: no input.rsg file found, defaults taken'
CJEM  777  CLOSE(23)

C Input some parameters from keyboard
C Input lower limit for gammas used in the extraction
      igmin=INT(((Eg_min-a0)/a1)+0.5)
      Eg_min=a0+a1*igmin
CJEM      WRITE(6,10)Eg_min
CJEM 10   FORMAT('Lower limit of gamma energy (keV)      <',F7.1,'>:',$)
CJEM     CALL READF(5,Eg_min)

      igmin=INT(((Eg_min-a0)/a1)+0.5)
      Eg_min=a0+a1*igmin
      jmin=INT(((Ex_min-a0)/a1)+0.5)
      jmin=MAX(jmin,igmin)
      Ex_min=a0+a1*jmin
CJEM      WRITE(6,11)Ex_min
CJEM 11   FORMAT('Lower limit of excitation energy (keV) <',F7.1,'>:',$)
CJEM      CALL READF(5,Ex_min)
      jmin=INT(((Ex_min-a0)/a1)+0.5)
      Ex_min=a0+a1*jmin
      IF(Ex_min.LT.Eg_min)THEN
         WRITE(6,'(A29)')'Warning, Ex_min<Eg_min not allowed, reset'
         Ex_min=Eg_min
      ENDIF

CJEM       jmax=INT(((Ex_max-a0)/a1)+0.5)
CJEM       jmax =MIN(jmax,YDIM)
CJEM       Ex_max=a0+a1*jmax
CJEM      WRITE(6,12)Ex_max
CJEM 12   FORMAT('Upper limit of excitation energy (keV) <',F7.1,'>:',$)
CJEM      CALL READF(5,Ex_max)
      jmax=INT(((Ex_max-a0)/a1)+0.5)
      jmax=MIN(jmax,YDIM)
      Ex_max=a0+a1*jmax
      IF(jmax.GT.511)THEN
         WRITE(6,13)Ex_max,a1
 13      FORMAT('Warning, Ex_max/a1 = ',F7.1,'/',F5.1,' > 511 channels,
     & reset')
         jmax=511
      ENDIF
      Emr=Ex_max-Eg_min
      iemr=INT(((Emr-a0)/a1)+0.5)
      Emr=a0+a1*iemr


      imax = jmax + iu0


      Eg_max=egmax(imax)
      IF(Eg_max.GT.Eg_limit)Eg_max = Eg_limit
      DO ix=jmin,jmax
         IF(igmax(ix).GT.imax)igmax(ix)=imax
c            write(6,*)'ix,igmax(ix)',ix,igmax(ix)
      ENDDO


C Calculating number of points in the matrix to be fitted
      degrees = 0
      DO ix=jmin,jmax
        degrees = degrees + (igmax(ix) - igmin)
      ENDDO
      write(6,*)'Number of data points   ',degrees
C Minus number of data points in the fit functions (rho and sigma)
      degrees=degrees-(jmax-jmin+1) - (igmax(jmax)-igmin+1)
      write(6,*)'DOF, data points-rho-sig',degrees

      WRITE(6,14)a0,a1,imax+1,jmax+1,Ex_min,Ex_max,a0,Emr,Eg_min,Eg_max
 14   FORMAT('Common calibration is a0=',F7.2,'keV and a1=',F7.2,
     &'keV/ch',
     + /,'Dimension is',I4,' x',I4, 
     + /,'excitation energy region is ',F6.0,'keV to ',F7.0,'keV',
     + /,'rho extracted from ',F6.0,'keV to ',F7.0,'keV',
     + /,'sig extracted from ',F6.0,'keV to ',F7.0,'keV')
C Finding number of counts in Fg(ig,ix) for each Ex 
      DO ix=jmin,jmax
         SumFg(ix)=0.
C          write(6,*)'igmin=',igmin,'igmax(ix)=',igmax(ix)
         DO ig=igmin,igmax(ix)
            SumFg(ix)=SumFg(ix)+Fg(ig,ix)
         ENDDO
C          write(6,*)'SumFg(',ix,')=',SumFg(ix) 
         DO ig=igmin,igmax(ix)
            IF(SumFg(ix).LE.0.)THEN
               FgN(ig,ix)=0.
               sFg(ig,ix)=0.
            ELSE
               FgN(ig,ix)=Fg(ig,ix)/SumFg(ix)
CJEM                write(6,*)'FgN(',ig,ix,')=',FgN(ig,ix)
CJEM                sFgN(ig,ix)=sFg(ig,ix)/SumFg(ix) ! Normalizing to Fg counts also for variance
            ENDIF
         ENDDO
      ENDDO
C Start value for Rho and Sig
      DO iu=0,jmax-igmin + iu0
         Rho(0,iu)=1.
C          write(6,*)'Rho(',iu,')=',Rho(0,iu)
      ENDDO
      DO ig=igmin,igmax(jmax)
         Sig(0,ig)=0.
         DO ix=MAX(jmin,2*ig-igmax(ig)),jmax  !We are adding for one ig along ix. But when ig>ix
            Sig(0,ig)=Sig(0,ig)+FgN(ig,ix)    !we start from 2*ig-igmax(ix)
         ENDDO
C        write(6,*)'Sig(',ig,')=',Sig(0,ig)
      ENDDO
 
C Statistical errors
C C Calculating statistical error of first generation spectra
      DO ix=jmin,jmax
         Ex=a0+a1*ix
         sFmax=0.
         isFEg=0
         TotM=MAX(1.,gM(Ex))            !Number of gammas for 1.gen. 
         BckM=MAX(0.,gM(Ex)-1.)         !Number og gammas from 2.+3.+... gen.
         DO ig=igmin,igmax(ix)
C Fg=total-background NaI spectra, factor 2 due to unfolding
C Total and background error are assumed to be independent from each 
C other, therefore we use SQRT(tot+bck) instead of SQRT(tot)+SQRT(bck)
            sFg(ig,ix)=2.*SQRT((TotM+BckM)*Fg(ig,ix))
            IF(sFg(ig,ix).GT.sFmax)THEN
               isFEg=ig
               sFmax=sFg(ig,ix)
            ENDIF
         ENDDO
C The factor 2 due to unfolding is very uncertain, it could be anything...
C We treat sFg(ig,ix) from Eg=Eg_min up to gamma energy (isFEg) for maximum 
C uncertainty (sFmax) in a special way (we have high sFg around Eg=0)
C The Factor 1 is again very uncertain
         IF(isFEg.GT.igmin)THEN
            DO ig=igmin,isFEg
               sFg(ig,ix)=sFmax*(1.+1.*(FLOAT(isFEg)-FLOAT(ig))
     &            /FLOAT(isFEg))
            ENDDO
         ENDIF
C As a further twist, we will smooth the errors (one-dimensionally) for each
C first generation spectrum. The smoothing is done over gamma energy intervals 
C of 3 MeV. This is new for version 1.2 of rhosigchi. Smoothing is performed 
C on the tmp() array.
         DO ig=igmin,igmax(ix)
            tmp(ig)=MAX(2.,sFg(ig,ix))  ! no errors less than 2 counts
            IF(Fg(ig,ix).GT.0.)THEN
                xx=ABS(tmp(ig)/Fg(ig,ix))
                IF(xx.LT.0.10)tmp(ig)=0.10*Fg(ig,ix) ! no errors less than 10 percent (25.01.2007)
            ENDIF
         ENDDO
         ism=NINT(300./a1)! +/-ism defines channel fit-region. Corrected 1500->300 11 feb 2016

C The smoothing procedure of Andreas starts, fasten seatbelts...
         DO ig=igmin,igmax(ix)
            istart=ig-ism
            istop=ig+ism
            ll=0
            IF(istart.LT.igmin)istart=igmin
            IF(istop.GT.igmax(ix))istop=igmax(ix) !corrected 11 feb 2016
            DO j=istart,istop
               ll=ll+1
               xfit(ll)=FLOAT(j)
               yfit(ll)=tmp(j)
            ENDDO
            IF(ll.GT.3)THEN
               y0=0.
               y1=0.
               y2=0.
               x0=0.
               x1=0.
               x2=0.
               x3=0.
               x4=0.
               DO j=1,ll
                  y0=y0+yfit(j)
                  y1=y1+yfit(j)*xfit(j)
                  y2=y2+yfit(j)*xfit(j)**2.
                  x0=x0+1.
                  x1=x1+xfit(j)
                  x2=x2+xfit(j)**2.
                  x3=x3+xfit(j)**3.
                  x4=x4+xfit(j)**4.
               ENDDO
               d0=x4*x2*x0+x3*x2*x1+x3*x2*x1-x4*x1*x1-x3*x3*x0-x2*x2*x2
               d1=y2*x2*x0+y1*x2*x1+y0*x3*x1-y2*x1*x1-y1*x3*x0-y0*x2*x2
               d2=y2*x2*x1+y1*x4*x0+y0*x3*x2-y2*x3*x0-y1*x2*x2-y0*x4*x1
               d3=y2*x3*x1+y1*x3*x2+y0*x4*x2-y2*x2*x2-y1*x4*x1-y0*x3*x3
               IF(d0.NE.0.)THEN
                  coef(1)=d1/d0
                  coef(2)=d2/d0
                  coef(3)=d3/d0
                  sFg(ig,ix)=coef(1)*FLOAT(ig)**2.+coef(2)*FLOAT(ig)+
     & coef(3)
               ENDIF
            ENDIF
         ENDDO
C The first generation spectrum at high gamma energies is very uncertain
C due to unfolding. Usually we produce a lot of high energy gamma counts
C in the unfolding procedure. Some channel might nevertheless turn out to
C contain almost zero counts (a factor of 250 less counts than neighbouring 
C channels has been observed). Still these channels should have comparable
C errors to their neighbouring channels. This will be taken into account in 
C the following: From gamma energies greater than Ex_min, the error in the 
C first-generation spectrum must not change by more than +/-30% per increasing 
C gamma bin. No errors are less than 2 counts.
         DO ig=igmin+1,igmax(ix)
            sFg(ig,ix)=MAX(0.7*sFg(ig-1,ix),sFg(ig,ix)) !to avoid large decrease in error
            sFg(ig,ix)=MIN(1.3*sFg(ig-1,ix),sFg(ig,ix)) !to avoid large increase in error
            sFg(ig,ix)=MAX(2.,sFg(ig,ix))               !to avoid too small errors
         ENDDO
      ENDDO
C Normalizing sFg(Eg,Ex)
      DO ix=jmin,jmax
         DO ig=igmin,igmax(ix)
            IF(SumFg(ix).LE.0.)THEN
               sFgN(ig,ix)=0.
            ELSE
               sFgN(ig,ix)=sFg(ig,ix)/SumFg(ix)
C                write(6,*)'sFgN(',ig,ix,')=',sFgN(ig,ix)
            ENDIF
         ENDDO
      ENDDO


C Calculating number of degrees of freedom
C Number of data points fitted in first generation spectrum
c      degrees=(jmax-jmin+1)*(jmax+jmin-2*igmin+2)/2                !CHECK THIS
c           write(6,*)'Number of data points',degrees
C Minus number of data points in the fit functions (rho and sigma)
c      degrees=degrees-2*(jmax-igmin+1)
c           write(6,*)'DOF, data points - rho - sig',degrees


C Iteration starts here
      CALL Iteration(FgN,sFgN,Rho,Sig)
C Finding extracted average 1. gen. spectra FgTeo and Chi**2
      DO it=0,nit
         deg=degrees
         Chi(it)=0.
         DO ix=jmin,jmax
            SumFg(ix)=0.
            DO ig=igmin,igmax(ix)
               iu=ix-ig+iu0
               SumFg(ix)=SumFg(ix)+Sig(it,ig)*Rho(it,iu)
            ENDDO
            DO ig=igmin,igmax(ix)
               iu=ix-ig+iu0
               IF(SumFg(ix).GE.0.)THEN
                  FgTeo(ig,ix)=Sig(it,ig)*Rho(it,iu)/SumFg(ix)
               ELSE 
                  FgTeo(ig,ix)=0.
               ENDIF
               IF(sFgN(ig,ix).GT.0.)THEN
                  Chi(it)=Chi(it)+((FgTeo(ig,ix)-FgN(ig,ix))
     & /sFgN(ig,ix))**2.
               ELSE
                  deg=deg-1
C Minus one data point in the calculation of Chi^2
               ENDIF
            ENDDO 
         ENDDO
         IF(deg.GT.0)THEN
            Chi(it)=Chi(it)/FLOAT(deg)
         ELSE
            Chi(it)=0.
         ENDIF
      ENDDO
C Showing indicators for how well the iteration converged
      i1MeV=jmax/4
      i2MeV=jmax/2
      i3MeV=3*jmax/4
      e1=a0+FLOAT(i1MeV)*a1
      e2=a0+FLOAT(i2MeV)*a1
      e3=a0+FLOAT(i3MeV)*a1
      step=nit/5
      WRITE(6,*)'______________________________________________________'
      WRITE(6,15)
 15   FORMAT('        Convergence test using various indicators')
      WRITE(6,16)(it,it=step,nit,step)
 16   FORMAT('Indicator Iteration = 0',5I6)
      IF(Rho(0,i1MeV).NE.0.)WRITE(6,17)e1,(Rho(it,i1MeV)/Rho(0,i1MeV),
     & it=0,nit,step)
      IF(Rho(0,i2MeV).NE.0.)WRITE(6,17)e2,(Rho(it,i2MeV)/Rho(0,i2MeV),
     & it=0,nit,step)
      IF(Rho(0,i3MeV).NE.0.)WRITE(6,17)e3,(Rho(it,i3MeV)/Rho(0,i3MeV),
     & it=0,nit,step)
 17   FORMAT('Rho/Rho0 at  U=',F5.0,6F6.2)
      IF(Sig(0,i1MeV).NE.0.)WRITE(6,18)e1,(Sig(it,i1MeV)/Sig(0,i1MeV),
     & it=0,nit,step)
      IF(Sig(0,i2MeV).NE.0.)WRITE(6,18)e2,(Sig(it,i2MeV)/Sig(0,i2MeV),
     & it=0,nit,step)
      IF(Sig(0,i3MeV).NE.0.)WRITE(6,18)e3,(Sig(it,i3MeV)/Sig(0,i3MeV),
     & it=0,nit,step)
 18   FORMAT('Sig/Sig0 at Eg=',F5.0,6F6.2)
      WRITE(6,19)(Chi(it),it=0,nit,step)
 19   FORMAT('Chi^2 for 1.gen.sp. ',6F6.2)
      WRITE(6,*)'______________________________________________________'
C Error calculation
C Initializing new Rhov = 1
      DO iu=0,jmax-igmin+iu0
         Rhov(0,iu)=Rho(0,iu)
      ENDDO
C New iteration were we test the influence of Fg -> Fg + sFg
C One by one channel is changed and (Sig-Sigv)**2
C and (Rho-Rhov)**2 is summed up, and SigmaRho and SigmaF is found
C Initializing arrays
      r=rand(time())               !seeding
      DO iu=0,jmax-igmin+iu0
         sRho(iu)=0.
      ENDDO
      DO ig=igmin,igmax(imax)
         sSig(ig)=0.
      ENDDO
      DO i=1,nunc
         DO ix=jmin,jmax
            DO ig=igmin,igmax(ix)
               r=rand(0)
               z=Finvert(r)
               Fgv(ig,ix)=Fg(ig,ix)+z*sFg(ig,ix) !Adding uncertainty
               IF(Fgv(ig,ix).LT.0.)Fgv(ig,ix)=0. !Zeroing negative numbers
            ENDDO
         ENDDO
         DO ix=jmin,jmax   !Finding number of counts in Fgv(ig,ix) for each Ex
            SumFgv(ix)=0.
            DO ig=igmin,igmax(ix)
               SumFgv(ix)=SumFgv(ix)+Fgv(ig,ix) 
            ENDDO
C Normalizing Fgv(Eg,Ex)
            DO ig=igmin,igmax(ix)
               IF(SumFgv(ix).LE.0.)THEN
                  Fgv(ig,ix)=0.
               ELSE
                  Fgv(ig,ix)=Fgv(ig,ix)/SumFgv(ix)
               ENDIF
            ENDDO
         ENDDO
C Initializing new Sigv
         DO ig=igmin,igmax(jmax)
            Sigv(0,ig)=0.
            DO ix=MAX(jmin,ig),jmax
               Sigv(0,ig)=Sigv(0,ig)+Fgv(ig,ix)
            ENDDO
         ENDDO
         CALL Iteration(Fgv,sFgN,Rhov,Sigv) !Iterating Fgv
         DO iu=0,jmax-igmin+iu0
            sRho(iu)=sRho(iu)+(Rhov(nit,iu)-Rho(nit,iu))**2.
         ENDDO
         DO ig=igmin,igmax(jmax)
            sSig(ig)=sSig(ig)+(Sigv(nit,ig)-Sig(nit,ig))**2.
         ENDDO
C Begin calculating relative errors
         Sum=0.
         Sum2=0.
         in0=0
         DO iu=0,jmax-igmin+iu0
            IF(Rho(nit,iu).GT.0.AND.sRho(iu).GT.0.)THEN
               rRho(iu)=SQRT(sRho(iu)/FLOAT(i))/Rho(nit,iu)
               in0=in0+1
            ELSE
               rRho(iu)=0.
            ENDIF
            Sum=Sum+rRho(iu)
         ENDDO
         DO ig=igmin,igmax(jmax)
            IF(Sig(nit,ig).GT.0.AND.sSig(ig).GT.0.)THEN
               rSig(ig)=SQRT(sSig(ig)/FLOAT(i))/Sig(nit,ig)
               in0=in0+1
            ELSE
               rSig(ig)=0.
            ENDIF
            Sum=Sum+rSig(ig)
         ENDDO  
         IF(in0.GT.0)THEN
            Sum=Sum/FLOAT(in0)
         ELSE
            Sum=0.
         ENDIF
         DO iu=0,jmax-igmin+iu0
            IF(Rho(nit,iu).GT.0.)Sum2=Sum2+(rRho(iu)-Sum)**2.
         ENDDO
         DO ig=igmin,igmax(jmax)
            IF(Sig(nit,ig).GT.0.)Sum2=Sum2+(rSig(ig)-Sum)**2.
         ENDDO
C End calculating relative errors
      ENDDO


      DO iu=0,jmax-igmin+iu0
         sRho(iu)=SQRT(sRho(iu)/FLOAT(nunc))
      ENDDO
      DO ig=igmin,igmax(jmax)
         sSig(ig)=SQRT(sSig(ig)/FLOAT(nunc))
      ENDDO
C Output of the results
      WRITE(6,*)
      WRITE(6,50)
50    FORMAT(' ch   Ex(keV)     Rho(1/MeV)  dRho(1/MeV) (UNNORMALIZED)')
      DO iu=0,jmax-igmin+iu0
        WRITE(6,51)iu, a0+a1*FLOAT(iu), Rho(nit,iu), sRho(iu)
51      FORMAT( I5,  F8.1,  E14.3, E12.3)
      ENDDO
      WRITE(6,*)
      WRITE(6,52)
52    FORMAT(' ch   Eg(keV)         Sig        dSig (UNNORMALIZED)')
      DO ig=igmin,igmax(jmax)
        WRITE(6,53)ig, a0+a1*FLOAT(ig), Sig(nit,ig), sSig(ig)
53      FORMAT( I5,  F8.1,  E14.3, E12.3)
      ENDDO


CJEM  Addition: Putting final iteration of rho and sig in vectors for return:
      do i = 0, 100
         Rho_fin(i) = Rho(nit,i)
         Sig_fin(i) = Sig(nit,i)
      enddo


C Writing matrices  out for mama
C Writing matrices to files
C       cal(1,1,1,1)=a0
C       cal(1,1,1,2)=a1
C       cal(1,1,1,3)=0.
C       cal(1,1,2,1)=a0
C       cal(1,1,2,2)=a1
C       cal(1,1,2,3)=0.
C       XDIM = imax+1
C c      YDIM=dim
C       YDIM = jmax+1
C       DO ix=0,YDIM-1
C          DO ig=0,XDIM-1
C             rMAT(IDEST,ig,ix)=FgN(ig,ix)
C          ENDDO
C       ENDDO
C       outfile='fg.rsg'
C       comment='Observed first generation matrix'
C       OPEN(20,FILE=outfile,ACCESS='SEQUENTIAL',ERR=99)
C       CALL norw2dim(20,comment)
C       CLOSE(20)
C       WRITE(6,36)outfile
C  36   FORMAT('Observed first generation matrix written to file:  ',A11)
C       DO ix=0,YDIM-1
C          DO ig=0,XDIM-1
C             rMAT(IDEST,ig,ix)=sFgN(ig,ix)
C          ENDDO
C       ENDDO
C       outfile='fgerr.rsg'
C       comment='Estimated first generation error matrix'
C       OPEN(20,FILE=outfile,ACCESS='SEQUENTIAL',ERR=99)
C       CALL norw2dim(20,comment)
C       CLOSE(20)
C       WRITE(6,37)outfile
C  37   FORMAT('Estimated first gen. err. matrix written to file:  ',A11)
C       DO ix=0,YDIM-1
C          DO ig=0,XDIM-1
C             rMAT(IDEST,ig,ix)=FgTeo(ig,ix)
C          ENDDO
C       ENDDO
C       outfile='fgteo.rsg'
C       comment='Theoretical first generation matrix'
C       OPEN(20,FILE=outfile,ACCESS='SEQUENTIAL',ERR=99)
C       CALL norw2dim(20,comment)
C       CLOSE(20)
C       WRITE(6,38)outfile
C  38   FORMAT('Theoret. first generation matrix written to file:  ',A11)

C C Calculate relative level density and strength function
C C Writing spectra out for PAW
C       outfile='rhopaw.rsg'
C       OPEN(UNIT=20,FILE=outfile)
C       DO iu=0,jmax
C          WRITE(20,*)Rho(nit,iu)
C       ENDDO
C       DO iu=0,jmax
C          WRITE(20,*)sRho(iu)
C       ENDDO
C       CLOSE(20)
C       WRITE(6,39)outfile
C  39   FORMAT('Unnormalized Rho and dRho written to array file:  ',A11)

C       outfile='sigpaw.rsg'
C       OPEN(UNIT=20,FILE=outfile)
C       DO ig=0,imax
C          WRITE(20,*)Sig(nit,ig)
C       ENDDO
C       DO ig=0,imax
C          WRITE(20,*)sSig(ig)
C       ENDDO
C       CLOSE(20)
C       WRITE(6,40)outfile
C  40   FORMAT('Unnormalized T and dT     written to array file:  ',A11)

C C Writting spectra to mama matrices
C       cal(1,1,2,1)=0.
C       cal(1,1,2,2)=1.
C       cal(1,1,2,3)=0.

C       YDIM=2
C       XDIM = jmax+1
C       DO iu=0,XDIM-1
C          rMAT(IDEST,iu,0)=Rho(nit,iu)
C          rMAT(IDEST,iu,1)=sRho(iu)
C       ENDDO
C       outfile='rhosp.rsg'
C       comment='Unnormalized Rho and dRho'
C       OPEN(20,FILE=outfile,ACCESS='SEQUENTIAL',ERR=99)
C       CALL norw2dim(20,comment)
C       CLOSE(20)
C       WRITE(6,41)outfile
C  41   FORMAT('Unnormalized Rho and dRho written to mama file:    ',A11)

C       YDIM=2
C       XDIM=imax+1
C       DO ig=0,XDIM-1
C          rMAT(IDEST,ig,0)=Sig(nit,ig)
C          rMAT(IDEST,ig,1)=sSig(ig)
C       ENDDO
C       outfile='sigsp.rsg'
C       comment='Unnormalized T and dT'
C       OPEN(20,FILE=outfile,ACCESS='SEQUENTIAL',ERR=99)
C       CALL norw2dim(20,comment)
C       CLOSE(20)
C       WRITE(6,42)outfile
C  42   FORMAT('Unnormalized T and dT     written to mama file:    ',A11)

C       OPEN(23,FILE='input.rsg',ACCESS='SEQUENTIAL',ERR=888)
C       WRITE(23,*)Eg_min,Ex_min,Ex_max ! deleted writing mass, corrected 11 feb 2016
C       CLOSE(23)
C 888   CONTINUE

C       STOP
C  99   WRITE(6,*)'Could not open file for results and spectra'
C       STOP
      END

C -----------------------------------------------------------------------------
      FUNCTION gM(Ex)
C Gamma-Multiplicity as function of Ex (in keV)
      a=42.12/100.
      b=0.0467/100.
      c=-0.128868E-05/100.
      gM=0.
      IF(Ex.GT.0..AND.Ex.LT.10000.)gM=a+b*Ex+c*Ex*Ex
      IF(Ex.GE.10000.)gM=a+b*10000.+c*10000.*10000.
      IF(gM.LT.0.)gM=0.
      RETURN
      END
C -----------------------------------------------------------------------------

      SUBROUTINE ELASTIC(Fi,Ff,A0i,A1i,A0f,A1f,Di,Df)
C The most magnificent stretch- and compress-routine
C ever created by a human being. It is complicated, but works!
C The routine stretches or compresses spectrum from initial
C calibration (A0i,A1i) to final (A0f,A1f). The dimensions
C of the real spectra are Di and Df. First channel is 0, so
C that the spectra occupy (0:D-1) channels.
C August 1994, Oslo Cyclotron Laboratory, Magne Guttormsen
      INTEGER Di,Df
      DIMENSION Fi(0:Di-1),Ff(0:Df-1)
C Testing
      IF(A1i.EQ.0..OR.A1f.EQ.0.)RETURN
C Zeroing final spectrum
      DO i=0,Df-1
         Ff(i)=0.  
      ENDDO
C Case where no action is taken
      IF(A0i.EQ.A0f.AND.A1i.EQ.A1f)THEN
         DO i=0,MIN0(Di-1,Df-1)
            Ff(i)=Fi(i)
         ENDDO
      RETURN
      ENDIF
C Taking counts in initial spectrum and find where
C to put it in final spectrum. Then it is distributed on
C the channel(s). The loop goes through all the
C channels i of the initial spectrum
      IF(ABS(A1i/A1f).LT.2)THEN
         DO i=0,Di-1
            IF(Fi(i).EQ.0.)GO TO 97
            EiL=A0i+A1i*(i-0.5)      !Step 1.0 chs left and right
            EiH=A0i+A1i*(i+0.5)
            CHf1=(EiL-A0f)/A1f       !CHf1 and CHf2 define limits where
            CHf2=(EiH-A0f)/A1f       !to put the counts in final spc.
            CHlength=ABS(CHf1-CHf2)  !Number of channels (float)
            CountCH=Fi(i)/CHlength   !Number of counts pr.ch unit
            CHfL=CHf1
            CHfH=CHf2
            IF(CHfL.GT.CHfH)THEN
               CHfL=CHf2
               CHfH=CHf1
            ENDIF
            j1=INT(CHfL+0.5)
            j2=INT(CHfH+0.5)                     !filling with CHwidth*CountCH
            IF(j1.GE.Df.OR.j2.LT.0)GO TO 97
            nch=j2-j1+1
            IF(nch.EQ.1)THEN                !One channel to fill
               IF(j1.GE.0)Ff(j1)=Ff(j1)+Fi(i)
               GO TO 97
            ENDIF
            IF(nch.GT.1)THEN                !Two or three channels to fill
               Counts=CountCH*(FLOAT(j1)+0.5-CHfL) !Fraction of left channel
               IF(j1.GE.0)Ff(j1)=Ff(j1)+Counts       
               Counts=CountCH*(CHfH+0.5-j2) !Fraction of right channel
               IF(j2.LE.Df-1)Ff(j2)=Ff(j2)+Counts
               DO j=j1+1,j2-1               !Filling in for whole chs.
                  IF(j.GE.0.AND.j.LE.Df-1)Ff(j)=Ff(j)+CountCH
               ENDDO
            ENDIF
 97         CONTINUE
         ENDDO
      ELSE
C The counts will be distributed in the streching procedure as a triangle,
C which has its left and right tail overlaping with the center of the 
C next triangle. The heighth and basis of the triangle is called h and b:
C               x         x          x
C             x   x     x   x      x   x
C           x       x x       x   x      x
C         x          0          0          x
C       x          x   x      x  x           x
C     x          x       x  x      x           x
C   x          x           0         x           x
         b=2.0*A1i/A1f   !basis of triangle
         h=2.0/b         !height of triangle in order to get area=1
         alpha=h/(b/2.)  !slope of triangle tails

         DO i=0,Di-1
            IF(Fi(i).EQ.0.)GO TO 98
            EiL=A0i+A1i*(i-1.)      !Step 1.0 chs left and right
            EiH=A0i+A1i*(i+1.)
            CHf1=(EiL-A0f)/A1f      !CHf1 and CHf2 define limits where
            CHf2=(EiH-A0f)/A1f      !to put the counts in final spc.
            CHfL=CHf1
            CHfH=CHf2
            IF(CHfL.GT.CHfH)THEN
               CHfL=CHf2
               CHfH=CHf1
            ENDIF
            j1=INT(CHfL)+1
            j2=INT(CHfH)
            IF(j1.GE.Df.OR.j2.LT.0)GO TO 98
            w=0.
            DO j=j1,j2
               IF(j.LT.CHfL+(b/2.))THEN
                  w=alpha*(j-CHfL)                !up going slope
               ELSE 
                  w=h-alpha*(j-(CHfL+(b/2.)))     !down going slope
               ENDIF
               IF(w.LT.-0.1)WRITE(6,*)'Warning, weight w < 0 : ',w
               IF(w.LT.0.)w=0.
               IF(j.GE.0.AND.j.LE.Df-1)Ff(j)=Ff(j)+w*Fi(i)
            ENDDO
 98         CONTINUE
         ENDDO
      ENDIF
      END

C -----------------------------------------------------------------------------
      SUBROUTINE Iteration(FgN,sFgN,Rho,Sig)
c      COMMON/iter/jmin,imax,igmin,nit,igmax(0:511),iu0
      COMMON/iter/jmin,jmax,igmin,nit,igmax(0:511),iu0

C Ansatz=FgN(E_x,E_gamma)=rho(E_x-E_gamma)*sigma(E_gamma)
      REAL Rho(0:100,0:511),Sig(0:100,0:511)
      REAL FgN(0:511,0:511),sFgN(0:511,0:511)
      REAL fun1(0:511),fun2(0:511),fun3(0:511)
      REAL nom(0:511,0:511),denom(0:511,0:511)
      REAL up,down,var
      INTEGER ii,it,ix,ig,iu,iu0,jmin,jmax,igmin,nit,igmax
C Start iteration
      DO it=1,nit
         IF(it.LE.5)THEN
            var=1.2
         ELSE IF(it.LE.12)THEN
            var=1.1
         ELSE IF(it.LE.21)THEN
            var=1.05
         ELSE IF(it.LE.30)THEN
            var=1.025
         ELSE
            var=1.01
         ENDIF
C Calculating two matrices 'nom' and 'denom'
         DO ix=jmin,jmax
            fun1(ix)=0.
            fun2(ix)=0.
            fun3(ix)=0.
            DO ig=igmin,igmax(ix)
               iu=ix-ig+iu0
                 IF(iu.LT.0)  write(6,*)'Warning iu is negative, ix, 
     & ig, iu = ',ix,ig,iu
               fun1(ix)=fun1(ix)+Sig(it-1,ig)*Rho(it-1,iu)
               IF(sFgN(ig,ix).GT.0.)THEN
                  fun2(ix)=fun2(ix)+(Sig(it-1,ig)*Rho(it-1,iu)
     & /sFgN(ig,ix))**2.
                  fun3(ix)=fun3(ix)+Sig(it-1,ig) *Rho(it-1,iu)
     & *FgN(ig,ix)/(sFgN(ig,ix)**2.)
               ENDIF
            ENDDO
            IF(fun1(ix).GT.0.)THEN
               fun2(ix)=fun2(ix)/(fun1(ix)**3.)
               fun3(ix)=fun3(ix)/(fun1(ix)**2.)
            ELSE
               fun2(ix)=0.
               fun3(ix)=0.
            ENDIF
            DO ig=igmin,igmax(ix)
               IF((fun1(ix)*sFgN(ig,ix)).GT.0.)THEN
                  nom(ig,ix)=fun2(ix)-fun3(ix)+FgN(ig,ix)
     & /(fun1(ix)*(sFgN(ig,ix)**2.))
                  denom(ig,ix)=1./(fun1(ix)*sFgN(ig,ix))**2.
               ELSE
                  nom(ig,ix)=0.
                  denom(ig,ix)=0.
               ENDIF
            ENDDO
         ENDDO
C Varying a point of sigma
         DO ig=igmin,igmax(jmax)
            up=0.
            down=0.
            ii=MAX(jmin,ig)
            DO ix=ii,jmax
               iu=ix-ig+iu0
               up=up+Rho(it-1,iu)*nom(ig,ix)
               down=down+(Rho(it-1,iu)**2.)*denom(ig,ix)
                  if(iu.lt.0)write(6,*)'Warning negative iu'
            ENDDO
            IF(down.GT.0.)THEN
               IF((up/down).GT.(var*Sig(it-1,ig)))THEN
                  Sig(it,ig)=var*Sig(it-1,ig)
               ELSE IF((up/down).LT.(Sig(it-1,ig)/var))THEN
                  Sig(it,ig)=Sig(it-1,ig)/var
               ELSE
                  Sig(it,ig)=up/down
               ENDIF
            ELSE 
               Sig(it,ig)=0.
            ENDIF
         ENDDO
C          if(it.eq.1)write(6,*)'Sig(',ig,')=',Sig(it,ig)
C Varying a point of rho
         DO iu=0,jmax-igmin+iu0
            up=0.
            down=0.
            ii=MAX(jmin,iu)
            DO ix=ii,jmax
               ig=ix-iu+iu0
               up=up+Sig(it-1,ig)*nom(ig,ix)
               down=down+(Sig(it-1,ig)**2.)*denom(ig,ix)
                 if(ig.lt.0)write(6,*)'Warning negative ig'
            ENDDO
            IF(down.GT.0.)THEN
               IF((up/down).GT.(var*Rho(it-1,iu)))THEN
                  Rho(it,iu)=var*Rho(it-1,iu)
               ELSE IF((up/down).LT.(Rho(it-1,iu)/var))THEN
                  Rho(it,iu)=Rho(it-1,iu)/var
               ELSE
                  Rho(it,iu)=up/down
               ENDIF
            ELSE
               Rho(it,iu)=0.
            ENDIF
C             if(it.eq.1)write(6,*)'Rho(',iu,')=',Rho(it,iu)
         ENDDO
      ENDDO
      RETURN
      END
C -----------------------------------------------------------------------------
      FUNCTION Finvert(y0)
C Inverting the monoton increasing function r=F(z) -> z=Finv(r) 
C This means to find the z-value giving the value y0 
C The function F is the cummulative Gauss function F=1/2(1+erf(z/sqrt(2)))
      REAL xl,xh,yl,yh,x,y,y0
      xl =-3.0
      xh = 3.0
      x  = 0.0
      yl = 0.0
      yh = 1.0
      y  = 0.5
      DO WHILE(ABS(y-y0).GT.0.001)
         x=xl+(xh-xl)*(y0-yl)/(yh-yl)
         y=0.5*(1.+erf(x/1.414213562))
         IF(y.GT.y0)THEN
            yl=y
            xl=x
         ELSE
            yh=y
            xh=x
         ENDIF
      ENDDO
      Finvert=x
      RETURN
      END





