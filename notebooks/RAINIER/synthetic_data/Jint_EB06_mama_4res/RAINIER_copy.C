/********************* RAINIER l.kirsch 04/27/2017 start date *****************/
/********************* version 1.2.0:   02/14/2018 add table+usr def models****/
/********************* lekirsch@lbl.gov ***************************************/
//  ____________________________________________
// |* * * * * * * *|############################|
// | * * * * * * * |                            |
// |* * * * * * * *|############################|
// | * * * * * * * |                            |
// |* * * * * * * *|############################|
// | * * * * * * * |                            |
// |* * * * * * * *|############################|
// |~~~~~~~~~~~~~~~'                            |
// |############################################|
// |                                            |
// |############################################|
// |                                            |
// |############################################|
// '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
// Randomizer of Assorted Initial Nuclear Intensities and Emissions of Radiation
// Follow readme to run in bash
// A) via run_RAINIER.sh script
// B) direct; but need to copy the settings.h file to the RAINIER directory
// $ root RAINIER.C++

// doubles, ints, and bools marked with prescript "d", "n", and "b" respectively
// arrays marked with prescript "a"
// globals marked with "g_" are accessible after the run, so are the functions
// full lowercase named variables (i.e. no CamelCase) are index variables
// precompiler commands "#" make things run fast when unnecessary items are out
// - better than 40 "if" statements for every call
// - also makes src code clear as to what contributes and what doesn't
// - dormant code is error prone
// the term "spin" usually means "angular momentum" in this code

/////////////////////////////// Program Includes ///////////////////////////////
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <sstream>
#include <stdlib.h>
#include <iomanip>
#include <vector>
#include <numeric>
#include "math.h"
#include "TTimeStamp.h"
using namespace std;
#include "TRandom2.h"
// 2=Tausworthe is faster and smaller than 3=Mersenne Twister (MT19937)
// search and replace "TRandom2" to "TRandom3" to change PRNG
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TGraphErrors.h"
#include "TString.h"
#include "TFile.h"
#include "TMath.h"
#include <TROOT.h>
#include "TF1.h"
#include "TF2.h"
#include "TTree.h"
#include <vector>
// determine OS for briccs
#ifdef __linux__
char cbriccs[] = "briccs";
#endif
#ifdef __MACH__
char cbriccs[] = "briccsMac";
#endif
#ifdef _WIN32
char cbriccs[] = "BrIccS.exe"; // haven't done any windows testing yet
#endif

/////////////////////////// Settings & Parameters ///////////////////////////////
#include "settings.h" // all (or most) parameters for the simulations
string g_sRAINIERPath; // Path to RAINIER

///////////////////////// Discrete Input File //////////////////////////////////
double g_adDisEne[g_nDisLvlMax]; // discrete lvl energy
double g_adDisSp [g_nDisLvlMax]; // discrete lvl spin
int    g_anDisPar[g_nDisLvlMax]; // discrete lvl parity
double g_adDisT12[g_nDisLvlMax]; // discrete lvl half-life
int    g_anDisGam[g_nDisLvlMax]; // number of gammas known
int    g_anDisGamToLvl[g_nDisLvlMax][g_nDisLvlGamMax]; // daughter lvls
double g_adDisGamBR [g_nDisLvlMax][g_nDisLvlGamMax]; // daughter branching ratio
double g_adDisGamICC[g_nDisLvlMax][g_nDisLvlGamMax]; // daughter Alpha ICC
// could do these dynamically, but takes time to code

double g_dECrit; // trust lvl scheme up to this E, determined by g_nDisLvlMax
void ReadDisInputFile() {
  ifstream lvlFile;
  TString szFile = g_sRAINIERPath + "/levels/z" + TString::Format("%03d",g_nZ) + ".dat";
  lvlFile.open(szFile.Data());
  if (lvlFile.fail()) {cerr << "Level File could not be opened" << endl; exit(0);}

  string sNucSearchLine;
  string sChem;
  int nA, nZ, nLvlTot = 0;
  bool bFoundNuc = false;
  int nTries = 1e6, nTry = 0;
  while(!bFoundNuc && nTry < nTries) {
    nTry++;
    getline(lvlFile, sNucSearchLine); // Header
    istringstream issHeader(sNucSearchLine);
    issHeader >> sChem >> nA >> nZ >> nLvlTot;
    if(nA == g_nAMass && nZ == g_nZ)
      {bFoundNuc = true; cout << "Nucleus: " << endl << sNucSearchLine << endl;}
  }
  if(nLvlTot < 2) { cerr << "err: No levels, check z file" << endl; cin.get(); }

  for(int lvl=0; lvl<g_nDisLvlMax; lvl++) {
    string sLvlLine;
    getline(lvlFile,sLvlLine);
    istringstream issLvl(sLvlLine);
    int nLvl, nLvlPar, nLvlGam;
    double dLvlEner, dLvlSp, dLvlT12;
    issLvl >> nLvl >> dLvlEner >> dLvlSp >> nLvlPar >> dLvlT12 >> nLvlGam;
    nLvl--; // to match convention of ground state is lvl 0

    if(nLvlPar == -1) nLvlPar = 0; // 1=+, 0=- diff convention dicebox and talys
    if(nLvl != lvl || lvl > nLvlTot) cerr << "err: File mismatch" << endl;
    if(int(dLvlT12) == dLvlT12) {

      // sometimes no halflife meas
      nLvlGam = dLvlT12; // missing half-life
      dLvlT12 = 999;
    }

    g_adDisEne[lvl] = dLvlEner;
    g_adDisSp[lvl]  = dLvlSp;
    g_anDisPar[lvl] = nLvlPar;
    g_anDisGam[lvl] = nLvlGam;
    g_adDisT12[lvl] = dLvlT12 * 1e15; // fs

    double dTotBR = 0.0;
    for(int gam=0; gam<nLvlGam; gam++) {
      string sGamLine;
      getline(lvlFile,sGamLine);
      istringstream issGam(sGamLine);
      int nGamToLvl;
      double dGamBR, dGamICC, dGamE, dPg;
      issGam >> nGamToLvl >> dGamE >> dPg >> dGamBR >> dGamICC;
      nGamToLvl--; // to match convention of ground state is lvl 0

      dTotBR += dGamBR;
      g_anDisGamToLvl[lvl][gam] = nGamToLvl;
      g_adDisGamBR   [lvl][gam] = dGamBR;
      g_adDisGamICC  [lvl][gam] = dGamICC;
    } // gam

    // Renormalize so BR adds to 1.000
    for(int gam=0; gam<nLvlGam; gam++) {
      g_adDisGamBR[lvl][gam] /= dTotBR;
    } // gam
  } // lvl
  lvlFile.close();

  g_dECrit = g_adDisEne[g_nDisLvlMax-1];
  #ifdef bForceBinNum
  g_dConESpac = (g_dExIMax - g_dECrit) / double(g_nConEBin);
  #else
  g_nConEBin = int((g_dExIMax - g_dECrit) / g_dConESpac) + 1; // 1 beyond
  #endif
} // Read Input

void PrintDisLvl() {
  cout << "****** Discrete ******" << endl;
  for(int lvl=0; lvl<g_nDisLvlMax; lvl++) {
    // levels
    cout << lvl << ":\t " << g_adDisEne[lvl] << "   " << g_adDisSp[lvl]
      << (g_anDisPar[lvl]==1?"+":"-")  // careful dicebox flips parity
      << " ";
      if(g_adDisT12[lvl] < 1e9) cout << g_adDisT12[lvl] << " fs" << endl;
      else cout << "N/A" << endl; // lifetime not measured
    // gammas
    for(int gam=0; gam<g_anDisGam[lvl]; gam++) {
      cout << "   " << g_anDisGamToLvl[lvl][gam] << " : "
        << g_adDisGamBR[lvl][gam] << endl;
    } // gam
  } // lvl
} // Print Discrete

TH3D *g_h3PopDist;
#ifdef bExFullRxn
/////////////////////// TALYS Rxn Population File //////////////////////////////
void ReadPopFile() {
  cout << "Reading Population File" << endl;
  // copy the "Population of Z= 60 N= 84 (144Nd) before decay" section
  // from TALYS File:
  //   projectile p
  //   element nd
  //   mass 144
  //   energy 10
  //   outpopulation y
  //   bins 54
  //   maxlevelstar 16

  double adEx [g_nExPopI];
  double adPop[g_nExPopI][g_nSpPopIBin][2];

  ifstream filePop;
  filePop.open(popFile);
  if (filePop.fail()) {cerr << "PopFile could not be opened" << endl; exit(0);}
  string sLine;
  //ifdef bParPop_Equipar: header " bin    Ex    Popul.    J= 0.0    J= 1.0..."
  //else                   header " bin    Ex    Popul.    J= 0.0-   J= 0.0+ ..."
  getline(filePop,sLine); //header
  getline(filePop,sLine); // header "blank line"

  while( getline(filePop, sLine) ) {
    istringstream issPop(sLine);
    int nBin;
    double dPopTot, dEx;
    issPop >> nBin >> dEx >> dPopTot;
    adEx[nBin] = dEx; // iss cant input directly into array elements
    for(int s=0; s<g_nSpPopIBin; s++) {
      #ifdef bParPop_Equipar
        double dPop;
        issPop >> dPop;
        // for equal parity: "fake" that all are read as negative parity
        adPop[nBin][s][0] = 2*dPop; // this should also fix the talys 2x "bug"
        adPop[nBin][s][1] = 0;
      #else
        for(int par=0; par<2; par++) { // read distribution per parity
          double dPop;
          issPop >> dPop;
          adPop[nBin][s][par] = dPop;
        } // read par
      #endif //bParPop_Equipar
    } // read J
  } // read E

  // need arrays for TH3D constructor; different from TH2D
  // arrays give the lower-edges; therefor the size must be nbin+1
  const double abinPar[3]={0,1,2}; // parity bins; positive:1 and negative:0
  double abinSpPop[g_nSpPopIBin+1];
  for(int binJ=0; binJ<=g_nSpPopIBin; binJ++){
    abinSpPop[binJ] = binJ;
  }
  g_h3PopDist = new TH3D("h3PopDist","h3PopDist",
    g_nSpPopIBin, abinSpPop, g_nExPopI-1, adEx, 2, abinPar);

  for(int binE=1; binE<g_nExPopI; binE++) {
    for(int binJ=1; binJ<=g_nSpPopIBin; binJ++) {
      for(int binPar=1; binPar<=2; binPar++) {
        g_h3PopDist->SetBinContent(binJ,binE,binPar,adPop[binE-1][binJ-1][binPar-1]);
      } // assign par
    } // assign J
  } // assign E
} // ReadPopFile
#endif // talys input for Oslo Analysis

////////////////////////// Level Density ///////////////////////////////////////
double GetEff(double dEx) {
  #ifdef bLD_BSFG
  double dEff = dEx - g_dE1;
  #endif
  #ifdef bLD_CTM
  double dEff = dEx - g_dE0;
  #endif
  #ifdef bLD_UsrDef
  double dEff = dEx - g_dE0;
  #endif
  #ifdef bLD_Table
  double dEff = dEx - g_dDelta;
  #endif
  if(dEff < 0.0) dEff = 0.00000001;
  return dEff;
} // GetEff

double GetLDa(double dEx) { // TALYS 1.8 asymptotic dependence
  double dEff = GetEff(dEx);
  #ifdef bLDaConst
  return g_dLDa;
  #endif
  #ifdef bLDaEx
  return g_dLDaAsym * (1 + g_dShellDelW * (1 - exp(-g_dDampGam * dEff)) / dEff);
  #endif
  #ifdef bLD_Table
  return grLDa->Eval(dEx);
  #endif
} // GetLDa

double GetSpinCut2(double dEx) {
  #ifndef bLD_Table
  double dEff = GetEff(dEx);
  double dLDa = GetLDa(dEx);
  #endif

  #ifdef bJCut_VonEgidy05 // Von Egidy PRC72,044311(2005)
  double dSpinCut2 = 0.0146 * pow(g_nAMass, 5.0/3.0)
  * (1 + sqrt(1 + 4 * dLDa * dEff)) / (2 * dLDa);
  #endif

  #ifdef bJCut_SingPart // Gholami PRC75,044308(2007)
  double dSpinCut2 = 0.1461 * sqrt(dLDa * dEff) * pow(g_nAMass, 2.0/3.0);
  #endif

  #ifdef bJCut_RigidSph // Grimes PRC10(1974) 2373-2386
  double dSpinCut2 = 0.0145 * sqrt(dEff / dLDa) * pow(g_nAMass, 5.0/3.0);
  #endif

  #ifdef bJCut_VonEgidy09 // Von Egidy PRC80,054310(2009)
  // empirical fit to other data with only mass table parameters
  double dExPd = dEx - 0.5 * g_dDeuPair;
  if(dExPd < 0) dExPd = 0.000001;
  double dSpinCut2 = 0.391 * pow(g_nAMass, 0.675)
    * pow(dExPd,0.312);
  #endif

  #ifdef bJCut_TALYS // TALYS 1.8 default:
  double dSpinCutF2 = 0.01389 * pow(g_nAMass, 5.0/3.0) / g_dLDaAsym
    * sqrt(dLDa * dEff);

  double dEffSn = GetEff(g_dSn);
  double dLDaSn = GetLDa(g_dSn);
  double dSpinCutSn2 = 0.01389 * pow(g_nAMass, 5.0/3.0) / g_dLDaAsym
    * sqrt(dLDaSn * dEffSn);

  double dSpinCutd2 = pow(g_dSpinCutd,2);
  double dSpinCut2 = 0.0;
  if(dEx <= g_dEd) {
    dSpinCut2 = dSpinCutd2;
  } else if(dEx < g_dSn) {
    dSpinCut2 = dSpinCutd2 + (dEx - g_dEd) / (g_dSn - g_dEd)
      * (dSpinCutSn2 - dSpinCutd2);
  } else {
    dSpinCut2 = dSpinCutF2;
  } // dEx condition
  #endif

  #ifdef bLD_Table
  double dJCut = grJCut->Eval(dEx);
  double dSpinCut2 = pow(dJCut,2);
  #endif

  #ifdef bJCut_UsrDef // choose/add what you like, these are some I've found:
  // everyone and their mother seems to have a favorite spin cutoff model
  //double dSpinCut2 = pow(0.83 * pow(g_nAMass,0.26),2); // TALYS 1.8 global
  double dSpinCut2 = pow(0.98 * pow(g_nAMass,0.29),2); // DICEBOX CTM
  //double dSpinCut2 = 0.0888*sqrt(dLDa * dEff) * pow(g_nAMass,2/3.0); //DICEBOX
  #endif

  return dSpinCut2;
}

double GetDensityTot(double dEx) {
  #ifdef bLD_CTM // Constant Temperture Function Model
  double dEff = GetEff(dEx);
  double dEnDen = exp(dEff / g_dTemp) / g_dTemp;
  #endif
  #ifdef bLD_UsrDef
  double dEff = GetEff(dEx);
  double dEnDen = exp(dEff / g_dTemp) / g_dTemp;
  #endif
  #ifdef bLD_BSFG // Back shifted Fermi Gas
  double dEff = GetEff(dEx);
  double dLDa = GetLDa(dEx);
  double dSpinCut2 = GetSpinCut2(dEx);
  double dEnDen = 1.0 / (12.0 * sqrt(2) * sqrt(dSpinCut2) * pow(dLDa, 0.25)
    * pow(dEff, 5.0 / 4.0) ) * exp(2.0 * sqrt(dLDa * dEff) );
  #endif
  #ifdef bLD_Table // From external file
  double dEnDen = grRho->Eval(dEx);
  #endif
  return dEnDen;
}

double GetDensity(double dEx, double dSp, int nPar) {
  double dEnDen = GetDensityTot(dEx);
  double dSpinCut2 = GetSpinCut2(dEx);

  double dSpDen = (dSp + 0.5) * exp(-pow(dSp + 0.5, 2) / (2 * dSpinCut2))
   / dSpinCut2;

  #ifdef bPar_Equipar
  int nParity = nPar; // basically unused in this model
  double dParDen = 0.5;
  #endif
  #ifdef bPar_Edep
  // Al-Quraishi PRC67,015803(2003)
  double dParDen;
  double dExpTerm = 1.0 / (1.0 + exp( g_dParC * (dEx - g_dParD) ) );
  if(g_bIsEvenA) {
    if(nPar == 1){ // positive parity
      dParDen = 0.5 * (1.0 + dExpTerm ); // lot of positive states low E
    } else { // negative parity
      dParDen = 0.5 * (1.0 - dExpTerm ); // not many negative states low E
    } // +/-
  } else { // odd A
    if(nPar == 0){ // negative parity
      dParDen = 0.5 * (1.0 + dExpTerm );
    } else { // postive parity
      dParDen = 0.5 * (1.0 - dExpTerm );
    } // +/-
  }
  #endif

  double dDenEJP = dEnDen * dSpDen * dParDen;
  return dDenEJP;
} // plot with: TF2 *fDen2 = new TF2("fDen2","GetDensity(y,x,1)",0,9,0,16); fDen2->Draw("colz")

//////////////////// Build Nucleus /////////////////////////////////////////////
// spins marked with postscript "b" refer to "bin": necessary for half-int spins
// not to be confused with boolean type prescript "b"
// b bin:    0    1    2    3    4    ...
// sp int:   0    1    2    3    4    ...
// half-int: 0.5  1.5  2.5  3.5  4.5  ...

int g_nConLvlTot; // # lvl in constructed scheme. includes multiple in same bin
int g_nConMaxLvlBin; // the largest number of levels in an EJP bin

// * = memory to be dynamically allocated
double *g_adConExCen; // centroid energies of constructed bins
int *g_anConLvl; // number of levels in an EJP bin
int *g_anConCumul; // cumulative levels for random width seeding

int EJP(int ex, int spb, int par) { // index for Energy E, Spin J, Parity P
  return ex + spb * g_nConEBin + par * g_nConEBin * g_nConSpbMax;
} // EJP

double GetInBinE(int nReal, int nConEx, int nSpb, int nPar, int nLvlInBin) {
  TRandom2 ranInBinE(1 + // have seen issues with 0 seed
    nPar +
    nConEx    * 2 +
    nReal     * 2 * g_nConEBin +
    nSpb      * 2 * g_nConEBin * g_nReal +
    nLvlInBin * 2 * g_nConEBin * g_nReal * g_nConSpbMax);
  double dInBinE = g_adConExCen[nConEx];
  dInBinE += (-g_dConESpac / 2.0 + ranInBinE.Uniform(g_dConESpac));
  return dInBinE;
} // GetInBinE

const double g_dPi = 3.14159265359;
void BuildConstructed(int nReal) {
  // get rid of previous allocations
  cout << "Constructing Lvl Scheme" << endl;
  delete[] g_adConExCen;
  delete[] g_anConLvl;
  delete[] g_anConCumul;
  g_adConExCen = new double[g_nConEBin]; // center energy of the bin
  g_anConLvl   = new int   [g_nConEBin * g_nConSpbMax * 2];
  g_anConCumul = new int   [g_nConEBin * g_nConSpbMax * 2];

  TRandom2 ranLvlGen(1 + nReal);
  g_nConLvlTot = g_nDisLvlMax; // include discrete lvls below
  g_nConMaxLvlBin = 0; // max of any bin

  #ifdef bPoisson
  for(int ex=0; ex<g_nConEBin; ex++) {
    g_adConExCen[ex] = g_dECrit + (ex + 0.5) * g_dConESpac;
    // bottom of bin ex=0 equals g_dECrit: no gap or overlap with discrete
    // constructed in-bin-energies are discretized by gaussian seeds

    for(int spb=0; spb<g_nConSpbMax; spb++) {
      for(int par=0; par<2; par++) {

        double dSp; // integer or half integer determination for density
        if(g_bIsEvenA) dSp = spb; else dSp = spb + 0.5;

        double dAvgNumLev = g_dConESpac * GetDensity(g_adConExCen[ex],dSp,par);
        int nRanNumLvl = ranLvlGen.Poisson(dAvgNumLev); // integer poisson
        g_anConLvl[EJP(ex,spb,par)] = nRanNumLvl;
        if(nRanNumLvl > g_nConMaxLvlBin) g_nConMaxLvlBin = nRanNumLvl;

        g_nConLvlTot += g_anConLvl[EJP(ex,spb,par)];
        g_anConCumul[EJP(ex,spb,par)] = g_nConLvlTot;
      } // par
    } // sp bin
  } // ex
  #endif

  #ifdef bWigner
  for(int ex=0; ex<g_nConEBin; ex++) { // assign level energies, same as above
    g_adConExCen[ex] = g_dECrit + (ex + 0.5) * g_dConESpac;
  } // ex

  for(int par=0; par<2; par++) {
    for(int spb=0; spb<g_nConSpbMax; spb++) {
      double dSp; // integer or half integer, for density
      if(g_bIsEvenA) dSp = spb; else dSp = spb + 0.5;

      ///// initialize E bins to 0 /////
      for(int ex=0; ex<g_nConEBin; ex++) {
        g_anConLvl[EJP(ex,spb,par)] = 0;
      } // ex

      ///// expected cumulative constructed # of lvls /////
      // Each JP has independent average energy bin spacing according to
      // the density inverse which is itself a function of energy
      double adExpCumulCon[g_nConEBin] = {0.0};
      for(int ex=0; ex<g_nConEBin; ex++) {
        double dEx = g_adConExCen[ex];
        if(ex == 0)
          adExpCumulCon[ex] = GetDensity(dEx, dSp, par) * g_dConESpac;
        else // ex != 0
          adExpCumulCon[ex] = GetDensity(dEx, dSp, par) * g_dConESpac
            + adExpCumulCon[ex-1];
      } // ex

      ///// Level assignment /////
      double dWigSampleSum = 0.0;
      // expectation value of avg dist between neighboring levels is 1
      for(int ex=0; ex<g_nConEBin; ex++) {
        while( dWigSampleSum < adExpCumulCon[ex] ) {
          dWigSampleSum += 2.0 / sqrt(g_dPi)
            * sqrt( -log( ranLvlGen.Uniform(1.0) ) );
          g_anConLvl[EJP(ex,spb,par)]++;
        } // WigSampleSum < ExpCumulCon
        g_nConLvlTot += g_anConLvl[EJP(ex,spb,par)];
        g_anConCumul[EJP(ex,spb,par)] = g_nConLvlTot;
      } // ex

    } // spb
  } // par
  #endif
} // BuildConstructed

void PrintConLvl() {
  int nSpbPrint = 9; // won't algin with double digit spins or bin content > 9
  cout << "****** Constructed ******" << endl;
  cout << "More levels exist at higher spins" << endl;
  cout << "Parity   ";
  for(int spb=0; spb<=nSpbPrint; spb++) { cout << "-" << " ";} cout << " ";
  for(int spb=0; spb<=nSpbPrint; spb++) { cout << "+" << " ";} cout << endl;
  cout << "Spin Bin ";
  for(int spb=nSpbPrint; spb>=0; spb--) { cout << spb << " ";} cout << " ";
  for(int spb=0; spb<=nSpbPrint; spb++) { cout << spb << " ";} cout << endl;

  cout << "E(MeV)   " << endl;
  for(int ex=0; ex<g_nConEBin; ex++) {
    cout << fixed << setprecision(3) << g_adConExCen[ex] << "    ";
    int par = 0;
    for(int spb=nSpbPrint; spb>=0; spb--) {
      cout << g_anConLvl[EJP(ex,spb,par)] << "|";
    } // sp bin
    cout << " ";
    par = 1;
    for(int spb=0; spb<=nSpbPrint; spb++) {
      cout << g_anConLvl[EJP(ex,spb,par)] << "|";
    } // sp bin
    // printing energy of every level in bin would be so many more lines
    // but could be done with something like:
    // for(int lvl=0; lvl<g_anConLvl[EJP(ex,spb,par)]; lvl++)
    //   cout << GetInBinE(ex,spb,par,lvl) << endl;
    cout << scientific << endl;
  } // ex
  cout << "Total Number of Levels = " << g_nConLvlTot << endl;
} // PrintConLvl

/////////////////// Transistion Type ///////////////////////////////////////////
int GetTransType(int nSpbI, int nParI, int nSpbF, int nParF) {
  // only integer spins
  int nTransType = 0;
    // 0 No trans; 1 Pure E1; 2 Mixed M1+E2; 3 Pure M1; 4 Pure E2
  int ndSpb = TMath::Abs(nSpbF - nSpbI);
  int ndPar = TMath::Abs(nParF - nParI);

  if(nSpbF < 0 || nSpbF >= g_nConSpbMax) return 0; // not possible
  if(nSpbI < 0 ) cerr << "err: Negative input spins" << endl;

  // there are faster ways to compute transistion type than what is below,
  // but the function is not called that many times
  // so its better to be fully explicit for clarity - trust me u dont want gotos
  // the only real head-scratchers occur when spin bin 0 is involved

  if(g_bIsEvenA) {
    if(ndSpb == 0) {
      if(nSpbF > 0 && nSpbI > 0) {
        if(ndPar == 0) {
          nTransType = 2; // M1+E2
        } else {
          nTransType = 1; // E1
        }
      } else {
        nTransType = 0; // no 0 -> 0 with gammas, ignore E0 Internal Conversion
      }
    } else if(ndSpb == 1) {
      if(nSpbF > 0 && nSpbI > 0) { // triangle check
        if(ndPar == 0) {
          nTransType = 2; // M1+E2
        } else {
          nTransType = 1; // E1
        }
      } else {
        if(ndPar == 0) { // 0+ -> 1+ or 0- -> 1- or 1- -> 0- or 1+ -> 0+
          nTransType = 3; // M1 Pure
        } else { // 0+ -> 1- or 0- -> 1+ or 1+ -> 0- or 1- -> 0+
          nTransType = 1; // E1
        }
      }
    } else if(ndSpb == 2) {
      if(ndPar == 0) {
        nTransType = 4; // E2
      } else {
        nTransType = 0; // no M2,E3
      }
    } else { // ndSpb > 2
      nTransType = 0; // no Octupole
    }
  } else { //////////////////// odd A //////////////////////////////////////////
    // 0 No trans; 1 Pure E1; 2 Mixed M1+E2; 3 Pure M1; 4 Pure E2
    if(ndSpb == 0) {
      if(nSpbF > 0 && nSpbI > 0) {
        if(ndPar == 0) {
          nTransType = 2; // M1+E2
        } else {
          nTransType = 1; // E1
        }
      } else {
        if(ndPar == 0) {
          nTransType = 3; // 1/2+ -> 1/2+ pure M1, no E2 via triangle condition
        } else {
          nTransType = 1; // 1/2+ -> 1/2- E1
        }
      }
    } else if(ndSpb == 1) {
      if(nSpbF > 0 && nSpbI > 0) { // triangle check
        if(ndPar == 0) {
          nTransType = 2; // M1+E2
        } else {
          nTransType = 1; // E1
        }
      } else {
        if(ndPar == 0) { // 1/2+ -> 3/2+ can be quadrupole unlike 0+ -> 1-
          nTransType = 2; // M1+E2
        } else { // 1/2+ -> 3/2- , etc.
          nTransType = 1; // E1
        }
      }
    } else if(ndSpb == 2) {
      if(ndPar == 0) {
        nTransType = 4; // E2
      } else {
        nTransType = 0; // no M2,E3
      }
    } else { // ndSpb > 2
      nTransType = 0; // no Octupole
    }
  } // A determination

  return nTransType;
} // GetTransType

/////////////////////// Gamma Strength * Eg^(2L+1) /////////////////////////////
// Physical Constants
const double g_d4Pi2 = 39.4784176; // 4*pi^2
const double g_dKX2 = 5.204155555E-08; // mb^-1 MeV^-2;  = 1/(5*(pi*hbar*c)^2)

double GetTemp(double dEx) {
  double dEff = GetEff(dEx);
  double dLDa = GetLDa(dEx);
  #ifdef b_GenLor_CT
  return g_GenLor_CT;
  #endif
  return sqrt(dEff / dLDa);
} // GetTemp

double GetStrE1(double dEx, double dEg) {
  double dStr = 0.0;

  #ifdef bGSF_Table
  dStr = grGSF_E1->Eval(dEg);
  #else
  double dTemp = GetTemp(dEx - dEg);

  for(int set=0; set<g_nParE1; set++) { // sum over split dipoles in applicable
    double dGam = g_adGamE1[set] * (dEg*dEg + g_d4Pi2 * dTemp*dTemp)
      / (g_adEneE1[set]*g_adEneE1[set]); // energy dependent width

    #ifdef bE1_GenLor // Kopecky and Uhl Gen Lorentzian;
    #ifdef bE1_EGLO
    if(g_nAMass >= 148) { // enhanced Generalized Lorentzian
      double deo = 4.5; // reference energy
      double dk = 1 + 0.09 * pow(g_nAMass-148,2) * exp(-0.18 * (g_nAMass-148));
      double dChi = dk + (1- dk) * (dEg - deo) / (g_adEneE1[0] - deo);
      dGam *= dChi;
    } // A > 148
    #endif

    double dTerm1 = dEg * dGam
      / ( pow( (dEg*dEg - g_adEneE1[set]*g_adEneE1[set]),2) + pow(dEg*dGam,2) );
    double dTerm2 = 0.7 * g_adGamE1[set] * g_d4Pi2 * dTemp*dTemp
      / pow(g_adEneE1[set],5);
      // non zero limit, F=0.7 is the fermi liquid quasiparticle collision fact
    double dTerm = dTerm1 + dTerm2;
    #endif

    #ifdef bE1_KMF // Kadmenskij Markushev Furman
    double dTerm = 0.7 * g_adEneE1[set] * dGam
      / pow( dEg*dEg - g_adEneE1[set]*g_adEneE1[set], 2);
    #endif

    #ifdef bE1_KopChr // Kopecky Chrien
    double dTerm = dEg * dGam
      / ( pow( (dEg*dEg - g_adEneE1[set]*g_adEneE1[set]),2) + pow(dEg*dGam,2) );
    #endif

    #ifdef bE1_StdLor // Standard Lorentzian
    double dTerm = dEg * g_adGamE1[set]
      / ( pow(dEg*dEg - g_adEneE1[set]*g_adEneE1[set], 2)
      + pow(dEg*g_adGamE1[set], 2) );
    #endif

    #ifdef bE1_UsrDef // user defined; includes multiple resonances
    // edit as you please
    double dTerm = dEg * g_adGamE1[set]
      / ( pow(dEg*dEg - g_adEneE1[set]*g_adEneE1[set], 2)
      + pow(dEg*g_adGamE1[set], 2) );
    #endif

    dStr +=  g_dKX1 * g_adSigE1[set] * g_adGamE1[set] * dTerm;
  } // E1 parameter set
  #endif // bGSF_Table

  if(dStr < 0) {cerr << "err: Negative strength" << endl;}
  return dStr * pow(dEg,3); // Eg^(2L+1) so this in not formally gamma strength
} // GetStrE1

double GetStrM1(double dEg) {
  // Standard Lorentzian
  double dStr = 0.0;
  #ifdef bGSF_Table
  dStr = grGSF_M1->Eval(dEg);
  #else
  for(int set=0; set<g_nParM1; set++) {
    #ifdef bM1_StdLor
    dStr += g_dKX1 * g_adSigM1[set] * dEg * g_adGamM1[set]*g_adGamM1[set]
      / ( pow(dEg*dEg - g_adEneM1[set]*g_adEneM1[set], 2)
      + pow(dEg*g_adGamM1[set], 2) );
    #endif

    #ifdef bM1_UsrDef // user defined; includes multiple resonances
    // edit as you please
    dStr += g_dKX1 * g_adSigM1[set] * dEg * g_adGamM1[set]*g_adGamM1[set]
      / ( pow(dEg*dEg - g_adEneM1[set]*g_adEneM1[set], 2)
      + pow(dEg*g_adGamM1[set], 2) );
    #endif

  } // M1 parameter set

  #ifdef bM1StrUpbend
  double dUpbend = g_dUpbendM1Const * exp(-g_dUpbendM1Exp * dEg);
  dStr += dUpbend;
  #endif

  #ifdef bM1_SingPart
  dStr = g_dSpSigM1;
  #endif

  #endif // bGSF_Table
  if(dStr < 0) {cerr << "err: Negative strength" << endl;}
  return dStr * pow(dEg,3);
}

double GetStrE2(double dEg) {
  #ifdef bE2_StdLor
  // Standard Lorentzian
  double dStr = g_dKX2 * g_dSigE2 * g_dGamE2*g_dGamE2
    / (dEg * (pow(dEg*dEg - g_dEneE2*g_dEneE2,2) + pow(dEg*g_dGamE2,2)));
  // divide by Eg so units work out: TALYS formula units don't work
  #endif

  #ifdef bE2_UsrDef // user defined
  // edit as you please
  double dStr = g_dKX2 * g_dSigE2 * g_dGamE2*g_dGamE2
    / (dEg * (pow(dEg*dEg - g_dEneE2*g_dEneE2,2) + pow(dEg*g_dGamE2,2)));
  #endif

  #ifdef bE2_SingPart
  double dStr = g_dSpSigE2;
  #endif

  #ifdef bGSF_Table
  double dStr = grGSF_E2->Eval(dEg);
  #endif
  if(dStr < 0) {cerr << "err: Negative strength" << endl;}
  return dStr * pow(dEg,5);
}

double g_de = 2.7182818284590452353; // Euler's number
double g_dlam = 0.5; // need to convert Gamma dist to Chi2 dist
double GetRanChi2(TRandom2 &ran, double dnu) {
  // ROOT doesn't supply Chi-Squared Dist tied to a TRandom
  // http://pdg.lbl.gov/2013/reviews/rpp2013-rev-monte-carlo-techniques.pdf
  double dAcceptX = -1;
  bool bAccept = false;
  double dk = dnu / 2.0;
  if(dk == 1) { // exponential dist
    double du = ran.Uniform();
    bAccept = true;
    dAcceptX = -log(du);
  } else if(dk > 0 && dk < 1) { // pole at 0, could return 0 due to underflow
    double dv1 = (g_de + dk) / g_de;
    while(!bAccept) {
      double du1 = ran.Uniform();
      double du2 = ran.Uniform();
      double dv2 = dv1 * du1;
      if(dv2 <= 1) {
        double dx = pow(dv2, 1.0/dk);
        if(du2 <= exp(-dx) ) {
          bAccept = true;
          dAcceptX = dx;
        } // else restart with new u1, u2
      } else { // dv2 > 1
        double dx = -log((dv1 - dv2) / dk);
        if(du2 <= pow(dx, dk-1)) {
          bAccept = true;
          dAcceptX = dx;
        } // else restart with new u1, u2
      } // v2 condition
    } // accept
  } else if(dk > 1) { // closed like gaussian
    double dc = 3 * dk - 0.75;
    while(!bAccept) {
      double du1 = ran.Uniform();
      double dv1 = du1 * (1 - du1);
      double dv2 = (du1 - 0.5) * sqrt(dc / dv1);
      double dx = dk + dv2 - 1;
      if(dx > 0) {
        double du2 = ran.Uniform();
        double dv3 = 64 * dv1*dv1*dv1 * du2*du2;
        if(  (dv3 <= 1 - 2 * dv2*dv2 / dx)
          || (log(dv3) <= 2 * ( (dk -1) * log(dx / (dk - 1)) - dv2) ) ) {
          bAccept = true;
          dAcceptX = dx;
        } // v3 condtion
      } // x condtion
    } // accept
  } else {
    cerr << "err: negative chi2 degree freedom" << endl;
  } // k conditon
  if(dAcceptX < 0) cerr << "err: no ch2 random assigned" << endl;
  return dAcceptX / g_dlam;
} // GetRanChi2

double GetStr(double dEx, double dEg, int nTransType, double &dMixDelta2,
  TRandom2 &ranStr) {
  // returns sum_XL( Str_XL * Eg^(2L+1) )
  dMixDelta2 = 0.0;
  switch(nTransType) {
    case 0: return 0.0; // save some flops for impossible transistions
    case 1: { // int nX = 0, nL = 1;
              #ifdef bWFD_PTD
              double dGaus = ranStr.Gaus();
              double dFluct = dGaus * dGaus;
              #endif
              #ifdef bWFD_nu
              double dFluct = GetRanChi2(ranStr, g_dNu) / g_dNu;
              #endif
              #ifdef bWFD_Off
              double dFluct = 1.0;
              #endif
              return dFluct * GetStrE1(dEx, dEg);
            }
    case 2: {
              // both decay branches fluctuate independently
              // int nX = 1, nL = 1;
              #ifdef bWFD_PTD
              double dGausM1 = ranStr.Gaus();
              double dFluctM1 = dGausM1 * dGausM1;
              #endif
              #ifdef bWFD_nu
              double dFluctM1 = GetRanChi2(ranStr, g_dNu) / g_dNu;
              #endif
              #ifdef bWFD_Off
              double dFluctM1 = 1.0;
              #endif
              double dStrM1 = dFluctM1 * GetStrM1(dEg);

              // int nX = 0, nL = 2;
              #ifdef bWFD_PTD
              double dGausE2 = ranStr.Gaus();
              double dFluctE2 = dGausE2 * dGausE2;
              #endif
              #ifdef bWFD_nu
              double dFluctE2 = GetRanChi2(ranStr, g_dNu) / g_dNu;
              #endif
              #ifdef bWFD_Off
              double dFluctE2 = 1.0; // PT fluct off
              #endif
              double dStrE2 = dFluctE2 * GetStrE2(dEg);

              // delta = <I1||E2||I0> / <I1||M1||I0>; I0,1 = init,final w.f.
              //       = sqrt(GamE2 / GamM1)
              //       = sqrt(strE2 * Eg^5 / strM1 * Eg^3)
              dMixDelta2 = dStrE2 / dStrM1; // fact of Eg in GetStrXL
              return dStrM1 + dStrE2;
            }
    case 3: { // int nX = 1, nL = 1;
              #ifdef bWFD_PTD
              double dGaus = ranStr.Gaus();
              double dFluct = dGaus * dGaus;
              #endif
              #ifdef bWFD_nu
              double dFluct = GetRanChi2(ranStr, g_dNu) / g_dNu;
              #endif
              #ifdef bWFD_Off
              double dFluct = 1.0;
              #endif
              return dFluct * GetStrM1(dEg);
            }
    case 4: { // int nX = 0, nL = 2;
              #ifdef bWFD_PTD
              double dGaus = ranStr.Gaus();
              double dFluct = dGaus * dGaus;
              #endif
              #ifdef bWFD_nu
              double dFluct = GetRanChi2(ranStr, g_dNu) / g_dNu;
              #endif
              #ifdef bWFD_Off
              double dFluct = 1.0;
              #endif
              return dFluct * GetStrE2(dEg);
            }
    default : cerr << "err: Invaild strength type" << endl; return 0.0;
  } // TransType
} // GetStr

////////////////////// Internal Conversion /////////////////////////////////////
#ifdef bUseICC
double GetBrICC(double dEg, int nTransMade=1, double dMixDelta2=0.0) {
  int nSuccess = -7;
  int nReadLine = 0;  // BrIcc output changes based on input
  switch(nTransMade) {
    case 0: return 0.0; break;
    case 1: nSuccess = system(
      Form("%s/%s -Z %d -g %f -L E1 -w %s > oAlpha.briccs",
      g_sRAINIERPath.c_str(),cbriccs, g_nZ, dEg*1000, g_sBrIccModel) ); nReadLine = 8; break; // MeV
    case 2: nSuccess = system(
      Form("%s/%s -Z %d -g %f -L M1+E2 -d %f -w %s > oAlpha.briccs",
      g_sRAINIERPath.c_str(), cbriccs, g_nZ, dEg*1000, sqrt(dMixDelta2), g_sBrIccModel) );
      nReadLine = 11; break;
    case 3: nSuccess = system(
      Form("%s/%s -Z %d -g %f -L M1 -w %s > oAlpha.briccs",
      g_sRAINIERPath.c_str(), cbriccs, g_nZ, dEg*1000, g_sBrIccModel) ); nReadLine = 8; break;
    case 4: nSuccess = system(
      Form("%s/%s -Z %d -g %f -L E2 -w %s > oAlpha.briccs",
      g_sRAINIERPath.c_str(), cbriccs, g_nZ, dEg*1000, g_sBrIccModel) ); nReadLine = 8; break;
    default: cerr << "err: impossible transistion" << endl;
  } // transistion
  if(nSuccess) cerr << "err: BrIcc failure" << endl;
  ifstream fileAlpha("oAlpha.briccs");
  string sAlphaLine;
  for(int line=0; line<nReadLine; line++) { getline(fileAlpha,sAlphaLine); }
  getline(fileAlpha,sAlphaLine);
  double dAlpha;
  istringstream issAlpha(sAlphaLine);
  issAlpha >> dAlpha;
  return dAlpha;
} // GetBrICC

// unmixed conv coeff from BrICC, hist interpolation faster than graph
TH1D *g_hE1ICC;
TH1D *g_hM1ICC;
TH1D *g_hE2ICC;
// plot after run: g_hE1ICC->Draw()
void InitICC() {
  cout << "Initializing internal conversion coefficients: alphas" << endl;
  g_hE1ICC = new TH1D("g_hE1ICC","g_hE1ICC", g_nBinICC,g_dICCMin,g_dICCMax);
  g_hM1ICC = new TH1D("g_hM1ICC","g_hM1ICC", g_nBinICC,g_dICCMin,g_dICCMax);
  g_hE2ICC = new TH1D("g_hE2ICC","g_hE2ICC", g_nBinICC,g_dICCMin,g_dICCMax);
  for(int bin=1; bin<=g_nBinICC; bin++) {
    double dBinCenter = g_hE1ICC->GetBinCenter(bin);
    g_hE1ICC->SetBinContent(bin, GetBrICC(dBinCenter,1,0));
    g_hM1ICC->SetBinContent(bin, GetBrICC(dBinCenter,3,0));
    g_hE2ICC->SetBinContent(bin, GetBrICC(dBinCenter,4,0));
    cout << bin << " / " << g_nBinICC << "\r" << flush;
  } // bin
  int nSuccess = system("rm oAlpha.briccs");
  cout << endl;
} // InitICC
#endif

double GetICC(double dEg, int nTransMade=1, double dMixDelta2=0.0) {
  #ifdef bUseICC
  switch(nTransMade) {
    case 0: return 0; break;
    case 1: return g_hE1ICC->Interpolate(dEg); break;
    case 2: {
      double dM1 = g_hM1ICC->Interpolate(dEg);
      double dE2 = g_hE2ICC->Interpolate(dEg);
      return (dM1 + dMixDelta2 * dE2) / (1 + dMixDelta2); } break;
    case 3: return g_hM1ICC->Interpolate(dEg); break;
    case 4: return g_hE2ICC->Interpolate(dEg); break;
    default: cerr << "err: impossible transistion" << endl;
  }
  return 0.0;
  #else
  double dEg1 = dEg; // to avoid unused warnings
  int nTransMade1 = nTransMade;
  double dMixDelta21 = dMixDelta2;
  return 0.0;
  #endif
} // GetICC

///////////////// Calculate Widths /////////////////////////////////////////////
double GetWidth(int nExI, int nSpbI, int nParI, int nLvlInBinI, int nReal,
  double *adConWid, double *adDisWid, TRandom2 *arConState) {

  TRandom2 ranStr(1 + nReal + g_nReal * ( g_anConCumul[EJP(nExI,nSpbI,nParI)]
    - g_anConLvl[EJP(nExI,nSpbI,nParI)] + nLvlInBinI )); // seed with lvl num

  double dTotWid = 0.0;
  // should not calculate widths out of a discrete state
  // already have BR, discrete width not used in TakeStep
  if(nExI >= 0) { // in constructed scheme
    #ifdef bExSingle
    double dExI;
    if( nExI == (g_nConEBin - 1) )
      dExI = g_adConExCen[nExI] + 0.5 * g_dConESpac; // start top of 1st bin
    else
      dExI = g_adConExCen[nExI];
    #else
    //double dExI = g_adConExCen[nExI]; // fast, good approximation
    double dExI = GetInBinE(nReal, nExI, nSpbI, nParI, nLvlInBinI);
    #endif

    double dSpI;
    if(g_bIsEvenA){ dSpI = nSpbI; } else { dSpI = nSpbI + 0.5; }
    double dLvlSpac = 1.0 / GetDensity(dExI,dSpI,nParI);
    if(dLvlSpac <= 0){cerr << "err: Level spacing" << endl; dLvlSpac = 0.00001;}

    /////// to constructed scheme ///////
    for(int spb=nSpbI-2; spb<=nSpbI+2; spb++) { // dipole and quadrapole
      for(int par=0; par<2; par++) {
        int nTransType = GetTransType(nSpbI,nParI,spb,par);
        if(nTransType) { // possible
          for(int ex=0; ex<nExI; ex++) {
            // Could look at transisitions within same energy bin,
            // but so tiny effect. Would have to GetInBinE() of both in and
            // out state so that we don't accidentally go up in energy,

            int nLvlTrans = g_anConLvl[EJP(ex,spb,par)]; // #lvls in final bin
            if(nLvlTrans) { // might save some flops by ignoring empties

              double dExF = g_adConExCen[ex];
                // could get precise E but takes comp time, hardly effects str
              double dEg = dExI - dExF;

              double dStr = 0.0;
              arConState[EJP(ex,spb,par)] = ranStr; // want same rand #s
                // backing out same in-bin widths in TakeStep
              for(int outlvl=0; outlvl<nLvlTrans; outlvl++) {
                double dMixDelta2; // for ICC
                double dStrTmp = // need to get delta2
                  GetStr(dExI, dEg, nTransType, dMixDelta2, ranStr);
                dStr += dStrTmp * (1.0 + GetICC(dEg,nTransType,dMixDelta2));
              } // outlvl

              adConWid[EJP(ex,spb,par)] = dStr * dLvlSpac;
              dTotWid += dStr * dLvlSpac;
            } // final bin has lvls
          } // ex
        } // possible
      } // par
    } // sp bin

    /////// to discrete ///////
    for(int lvl=0; lvl<g_nDisLvlMax; lvl++) {
      double dExF = g_adDisEne[lvl];
      int nParF   = g_anDisPar[lvl];
      double dEg = dExI - dExF;

      double dSpF = g_adDisSp[lvl];
      int nSpbF; // convert half-int to int for binned transistion type
      if(g_bIsEvenA) nSpbF = int(dSpF + 0.001); else nSpbF = int(dSpF - 0.499);
        // protection against double imprecision
      int nTransType = GetTransType(nSpbI,nParI,nSpbF,nParF);
      adDisWid[lvl] = 0.0; // dont want uninitialized
      if(nTransType) { // possible
        double dMixDelta2; // for ICC
        double dStrTmp = // need to get delta2
          GetStr(dExI, dEg, nTransType, dMixDelta2, ranStr);
        double dStr = dStrTmp * (1.0 + GetICC(dEg,nTransType,dMixDelta2));

        adDisWid[lvl] = dStr * dLvlSpac;
        dTotWid += dStr * dLvlSpac;
      } // possible
    } // discrete lvl
  } else cerr << "err: Requesting discrete width" << endl;
  return dTotWid;
} // GetWidth

/////////////////////// Lifetime ///////////////////////////////////////////////
const double g_dHBar = 6.5821195e-7; // MeV fs
double GetDecayTime(double dTotWid, TRandom2 &ranEv) {
  double dLifeT = g_dHBar / dTotWid; // fs
  return ranEv.Exp(dLifeT);
}

////////////////////////// Take Step ///////////////////////////////////////////
bool TakeStep(int &nConEx, int &nSpb, int &nPar, int &nDisEx, int &nLvlInBin,
  int &nTransMade, double &dMixDelta2, double dTotWid, int nReal,
  double *adConWid, double *adDisWid, TRandom2 *arConState, TRandom2 &ranEv) {
  // TakeStep will change variables with &
  // Initial variables not marked with "I" because they are also the final state
  // use ConEx instead of nEx to be explicit that this is for constructed region
    // wasn't a problem in GetWidth since constructed region was prerequisite

  int nToConEx;
  int nToDisEx = g_nDisLvlMax;
  int nToSpb = -7, nToPar = -7, nToLvlInBin = -7; // junk initializers
  double dToMixDelta2 = 0.0;

  /////// in constructed scheme ///////
  if(nConEx >= 0) {
    double dSp;
    if(g_bIsEvenA){ dSp = nSpb; } else { dSp = nSpb + 0.5; }
    #ifdef bExSingle
    double dExI;
    if( nConEx == (g_nConEBin - 1) )
      dExI = g_adConExCen[nConEx] + 0.5 * g_dConESpac; // start top of 1st bin
    else
      dExI = g_adConExCen[nConEx];
    #else
    //double dExI = g_adConExCen[nExI];
    double dExI = GetInBinE(nReal, nConEx, nSpb, nPar, nLvlInBin);
    #endif

    if(dTotWid <= 0.0) return false; // dead state in constructed: no E2 options
    double dRanWid = ranEv.Uniform(dTotWid);
    double dWidCumulative = 0.0;
    bool bFoundLvl = false;

    ///// decay to discrete? /////
    for(int lvl=0; lvl<g_nDisLvlMax; lvl++) {
      dWidCumulative += adDisWid[lvl]; // already has ICC
      if(adDisWid[lvl] > 1e-4) cerr << "err: discrete width uninit" << endl;
      if(dWidCumulative >= dRanWid) {//once adds to more than dRanWid, it decays
        if(!bFoundLvl) { // for safety check
          bFoundLvl = true;
          nToConEx = -1;
          nToDisEx = lvl;
          if(g_bIsEvenA) nToSpb = int(g_adDisSp[lvl] + 0.001);
          else nToSpb = int(g_adDisSp[lvl] - 0.499);
          nToPar = g_anDisPar[lvl];
          nToLvlInBin = 0;
          nTransMade = GetTransType(nSpb,nPar,nToSpb,nToPar);
          lvl = g_nDisLvlMax; // break out of loop, for speed
        } // found lvl
      } // Cumulative >= Rand
    } // discrete lvl

    ///// decay to constructed scheme? /////
    for(int spb=nSpb-2; spb<=nSpb+2; spb++) { // dipole and quadrapole
      for(int par=0; par<2; par++) {
        int nTransType = GetTransType(nSpb,nPar,spb,par);
        if(nTransType != 0) {
          for(int ex=0; ex<nConEx; ex++) {
            double dConWid = adConWid[EJP(ex,spb,par)];
            if(dConWid > 1e-4) cerr << "err: con width uninit" << endl;
            dWidCumulative += dConWid;

            if(dWidCumulative >= dRanWid) {// once adds up to dRanWid, it decays
              if(!bFoundLvl) { // possibly already decayed to discrete
                bFoundLvl = true;
                nToConEx = ex;
                nToDisEx = g_nDisLvlMax;
                nToSpb = spb;
                nToPar = par;
                nTransMade = nTransType;
                // need to backtrack width sum and find out which individual
                // level in EJP bin it decayed to since many levels in a bin
                // each with random width according to PT distribution
                dWidCumulative -= dConWid;
                bool bFoundLvlInBin = false;

                double dLvlSpac = 1.0 / GetDensity(dExI,dSp,nPar);
                int nLvlTrans = g_anConLvl[EJP(ex,spb,par)];
                if(nLvlTrans) {
                  double dExF = g_adConExCen[ex];
                  double dEg = dExI - dExF;
                  TRandom2 ranStr = arConState[EJP(ex,spb,par)];

                  for(int outlvl=0; outlvl<nLvlTrans; outlvl++) {
                    double dMixDelta2Tmp;
                    double dStrTmp = // need to get delta2
                      GetStr(dExI, dEg, nTransType, dMixDelta2Tmp, ranStr);
                    double dStr = dStrTmp * (1.0 +
                      GetICC(dEg,nTransType,dMixDelta2Tmp));
                    dWidCumulative += dStr * dLvlSpac;
                    if(dWidCumulative >= dRanWid) {
                      if(!bFoundLvlInBin) { // possibly decayed prev inbin lvl
                        bFoundLvlInBin = true;
                        nToLvlInBin = outlvl;
                        dToMixDelta2 = dMixDelta2Tmp;
                        outlvl = nLvlTrans; // break loop for speed
                      } // found lvl in bin
                    } // Cumulative >= Rand
                  } // outlvl

                  ex = nConEx; // break out of loops, for speed
                  spb = g_nConSpbMax;
                  par = 2;
                } // final bin has lvls
              } // found lvl
            } // Cumulative >= Rand
          } // ex
        } // possible
      } // par
    } // sp bin
  } ///// in constructed scheme /////
  else { /////// in discrete ///////
    nToConEx = -1;
    double dRanBR = ranEv.Uniform(1.0); // should all add up to 1
    double dBRCumulative = 0.0;
    bool bFoundLvl = false;
    for(int gam=0; gam<g_anDisGam[nDisEx]; gam++) {
      dBRCumulative += g_adDisGamBR[nDisEx][gam]; // already has ICC in it
      if(dBRCumulative >= dRanBR) { // once adds to more than dRanBR, it decays
        if(!bFoundLvl) { // for safety check
          bFoundLvl = true;
          nToConEx = -1;
          nToDisEx = g_anDisGamToLvl[nDisEx][gam];
          if(g_bIsEvenA) nToSpb = int(g_adDisSp[nToDisEx] + 0.001);
          else nToSpb = int(g_adDisSp[nToDisEx] - 0.499);
          nToPar = g_anDisPar[nToDisEx];
          nToLvlInBin = 0;
          nTransMade = GetTransType(nSpb,nPar,nToSpb,nToPar);
          gam = g_anDisGam[nDisEx]; // break loop for speed
        } // found lvl
      } // Cumulative >= Rand
    } // gam choices
  } ///// in discrete /////

  // this is where user errors usually turn up the most
  if(nToSpb < 0 || nToPar < 0 || nToLvlInBin < 0 ) { // error check
    cerr << endl << "err: JP: from " << endl
         << nSpb << (nPar?"+":"-") << ", Lvl: " << nDisEx << ", ConBin: "
         << nConEx << ";" << nLvlInBin << endl << "To " << endl
         << nToSpb << (nToPar?"+":"-") << ", Lvl: " << nToDisEx << ", ConBin: "
         << nToConEx << ";" << nToLvlInBin << endl
         << "Likely branching ratios from file don't add to 1.000000" << endl
         << "Check level " << nDisEx << " in " << "zFile" << " manually."
         << endl;
         // else write code to normalize
    return false;
  } // err
  nConEx = nToConEx;
  nDisEx = nToDisEx;
  nSpb   = nToSpb;
  nPar   = nToPar;
  nLvlInBin = nToLvlInBin;
  dMixDelta2 = dToMixDelta2;
  return true;
} // TakeStep

///////////////// Get coninuum bin number for a given excitation energy ////////////
int GetContExBin(double dExcs) {
  for(int nExI=0; nExI<g_nConEBin; nExI++){
    if(abs(dExcs-g_adConExCen[nExI]) < g_dConESpac/2){ // if within +-1/2 bin width
      return nExI;
    }
  }
  return -1; // if no fitting bin was found
}

///////////////////////// Initial Excitation ///////////////////////////////////
void GetExI(int &nExI, int &nSpbI, int &nParI, int &nDisEx, int &nLvlInBinI,
  TRandom2 &ranEv, double dExIMean, double dExISpread) {
  // GetExI will change above variables marked with &

  ///// DICEBOX-like initial state /////
  #ifdef bExSingle
  // one starting state
  #ifdef bForceBinNum
  nExI = g_nConEBin - 1;
  #else
  double dExI = g_dExIMax;
  nExI = round( (dExI - g_dECrit) / g_dConESpac );
  #endif
  nSpbI = int(g_dSpI);
  nParI = g_dParI;
  nDisEx = g_nDisLvlMax;
  nLvlInBinI = 0;
  #endif

  ///// Beta-decay like selection of states /////
  #ifdef bExSelect
  double dRanState = ranEv.Uniform(1.0);
  double dBRSum = 0.0;
  for(int state=0; state<g_nStateI; state++) {
    dBRSum += g_adBRI[state];
    if(dBRSum > dRanState) {
      double dExI = g_adExI[state];
      nSpbI = int(g_adSpI[state]);
      nParI = g_anParI[state];

      // check wether level exists
      if(dExI < g_dECrit + 0.001) { // is discrete
        // need to be very careful of doublets and precision
        nExI = -1;
        bool bDisBinFound = false;
        for(int lvl=0; lvl<g_nDisLvlMax; lvl++) { // find discrete
          double dLvlE = g_adDisEne[lvl];
          if(TMath::Abs(dExI - dLvlE) < 0.001) {
            nDisEx = lvl;
            if (nSpbI == g_adDisSp[nDisEx] && nParI == g_anDisPar[nDisEx]) {
              bDisBinFound = true;
              break; // dis lvl
            }
          } // match E
        } // lvl
        if( !bDisBinFound ) cerr << "err: no discrete match @ " << dExI
                                 << " MeV, J/pi= " << nSpbI << "," << nParI << endl;
        if(g_bIsEvenA) nSpbI = int(g_adDisSp[nDisEx] + 0.001);
        else nSpbI = int(g_adDisSp[nDisEx] - 0.499);
        nParI = g_anDisPar[nDisEx];
        nLvlInBinI = 0;
        break; // state
      } // discrete
      else { // is continuum level
        nExI = round( (dExI - g_dECrit) / g_dConESpac );
        nDisEx = g_nDisLvlMax; //dummy
        // For now: assuming selection rules dJ = dPi = 0!
        int nLvlAvail = g_anConLvl[EJP(GetContExBin(dExI),nSpbI,nParI)];
        if(nLvlAvail>0) {
          nLvlInBinI = ranEv.Integer(nLvlAvail); // any lvls in bin fine
          break; // states
        }
        else {
          cerr << "\n" << "err: No level to populate for Ex=" << dExI
          << "\n check spin-parity of generated continuum level vs population from bExSelect"
          << endl;
        }
      } // continuum
    } // BR > RanState
  } // state
  #endif

  #ifdef bExSpread
  ///// Energy spread /////
  nDisEx = g_nDisLvlMax; // start with all discrete levels as possibilities
  bool bFoundSpin = false;
  nParI = ranEv.Integer(2);

  // populate J according to set distribution:
  double adJIPop[g_nConSpbMax];
  double dJIPopIntegral = 0.0;
  for(int spb=0; spb<g_nConSpbMax; spb++) {
    double dSp; // integer or half integer, for density
    if(g_bIsEvenA) dSp = spb; else dSp = spb + 0.5;

    // distribution:
    double dSpinCut2 = GetSpinCut2(dExIMean);
    double dSpDen = (dSp + 0.5) * exp(-pow(dSp + 0.5, 2) / (2 * dSpinCut2))
      / dSpinCut2;
    adJIPop[spb] = dSpDen;
    dJIPopIntegral += dSpDen;
  } // spb

  while(!bFoundSpin) {
    #ifdef bJIUnderlying
    double dRanJPop = ranEv.Uniform(dJIPopIntegral);
    double dJPopSum = 0.0;
    bool bJSuggested = false;

    for(int spb=0; spb<g_nConSpbMax; spb++) {
      double dJPop = adJIPop[spb];
      dJPopSum += dJPop; // populate if JPopSum > RanJPop
      if(!bJSuggested && (dJPopSum > dRanJPop) ) {
        bJSuggested = true;
        nSpbI = spb;
      } // JSuggested
    } // spb
    #endif

    #ifdef bJIPoisson
    nSpbI = ranEv.Poisson(g_dJIMean); // Poisson J dist
    #endif

    #ifdef bJIGaus
    nSpbI = ranEv.Gaus(g_dJIMean,g_dJIWid);
    if(nSpbI<0) nSpbI = 0; // positive gaussian
    #endif

    int nAttempt = 0;
    int nMaxAttempt = 1000;
    bool bFoundLvl = false;
    while(!bFoundLvl && nAttempt < nMaxAttempt){ // dont pop not existent lvls
      nAttempt++;
      nExI = round( (dExIMean - g_dECrit + ranEv.Gaus(0.0, dExISpread) )
        / g_dConESpac);
      if(nExI > g_nConEBin) cerr << "err: ExI above constructed max" << endl;
      if(nExI < 0) cerr << "err: ExI below constructed scheme" << endl;

      int nLvlAvail = g_anConLvl[EJP(nExI,nSpbI,nParI)];
      if(nLvlAvail > 0) {
        nLvlInBinI = ranEv.Integer(nLvlAvail);
        bFoundLvl = true;
        bFoundSpin = true;
      } // lvl avail
    } // attempts at finding a level at given spin

    // if is no lvl with the given spin in the level scheme, there is not
    // much more you can do than to repick a spin, throws off given spin dist

  } // found spin
  #endif // specific initial EJP range

  #ifdef bExFullRxn
  // Randomly selects EJP bin from input file population distribution
  // if no level in corresponding bin, searches nearby E bins
  // - not much more you can do when matching continuum and discrete physics
  double dSp = 0.0, dEx = 0.0;
  double dPopIntegral = g_h3PopDist->Integral(); // could calc outside
  bool bLvlMatch = false;

  // find a lvl from the histogram:
  bool bEJLvlSuggested = false;
  // Dont g_h2PopDist->GetRandom2(dSp, dEx)! ROOT's GetRandom2() is real crap!
  // Dont g_h3PopDist->GetRandom3(dSp, dEx, par)! ROOT's GetRandom2() is real crap!
  double dRanPop = ranEv.Uniform(dPopIntegral);
  double dPopSum = 0.0;
  for(int binE=1; binE<=g_nExPopI-1; binE++) {
    for(int binJ=1; binJ<=g_nSpPopIBin; binJ++) {
      for(int binPar=1; binPar<=2; binPar++) {
        // Does not do any interpolation of TALYS hist - room for improvement
        double dPop = g_h3PopDist->GetBinContent(binJ,binE,binPar);
        dPopSum += dPop; // populate if PopSum > RanPop
        if(!bEJLvlSuggested && (dPopSum > dRanPop) ) {
          bEJLvlSuggested = true;
          dSp = binJ-1; // 1st bin is spin 0 or 0.5
          double dLowEBdy = g_h3PopDist->GetYaxis()->GetBinLowEdge(binE);
          double dUpEBdy  = g_h3PopDist->GetYaxis()->GetBinUpEdge(binE);
          // matching the constructed and discrete regions aint pretty:
          if(dLowEBdy < g_dECrit + 0.001) { // is discrete
            // need to be very careful of doublets and precision
            nExI = -1;
            bool bDisBinFound = false;
            for(int lvl=0; lvl<g_nDisLvlMax; lvl++) { // find discrete
              double dLvlE = g_adDisEne[lvl];
              // cout << "dLvlE: " << dLvlE << endl;
              // cout << "dLowEBdy: " << dLowEBdy << endl;
              if(TMath::Abs(dLowEBdy - dLvlE) < 0.001) {
                nDisEx = lvl;
                bDisBinFound = true;
                bLvlMatch = true;
              } // match E
            } // lvl
            if( !bDisBinFound ) cerr << "err: no discrete match" << endl;
            if(g_bIsEvenA) nSpbI = int(g_adDisSp[nDisEx] + 0.001);
            else nSpbI = int(g_adDisSp[nDisEx] - 0.499);
            nParI = g_anDisPar[nDisEx];
            nLvlInBinI = 0;
          } else { // is constructed
            dEx = dLowEBdy + ranEv.Uniform(dUpEBdy - dLowEBdy);
            #ifdef bParPop_Equipar
              nParI = ranEv.Integer(2); // equal positive:1 and negative:0
            #else
              nParI = binPar-1; // selected through the loop
            #endif // bParPop_Equipar
            nSpbI = int(dSp); // should work for both even and odd A
            nExI = round( (dEx - g_dECrit) / g_dConESpac);
            if(nExI > g_nConEBin) cerr << "err: ExI above constructed max"
              << endl;
            if(nExI < 0) cerr << "err: ExI below constructed scheme" << endl;

            // search in bin then nearby
            bool bPlacedLvl = false;
            while(!bPlacedLvl) {
              int nLvlAvail = g_anConLvl[EJP(nExI,nSpbI,nParI)];
              if(nLvlAvail > 0) { // dont pop lvls that dont exist
                bLvlMatch = true;
                bPlacedLvl = true;
                nLvlInBinI = ranEv.Integer(nLvlAvail); // any lvls in bin fine
                nDisEx = g_nDisLvlMax; // start with all discrete lvls
              } else { // random walk in energy space, avoid ECrit line
                int nEBinStep = ranEv.Integer(2);
                if(nEBinStep) { // 0=down; 1=up
                  nExI++;
                  if(nExI >= g_nConEBin) { // stay below maximum energy
                    nExI -= 2;
                  } // ex max
                } else { // step down
                  nExI--;
                  if(nExI < 0 ) { // stay above ECrit
                    nExI += 2;
                  } // ex min
                } // increase or decrease E
              } // lvl avail
            } // placed lvl
          } // dis or con

        } // suggestion found

     } // bin par
    } // bin J
  } // bin E
  if(!bEJLvlSuggested) cerr << "err: lvl not suggested" << endl;
  if(!bLvlMatch) cerr << "err: lvl not found" << endl;
  #endif // large swath of EJP
} // GetExI

TF1 *fnLDa, *fnSpCut, *fnGSFE1, *fnGSFM1, *fnGSFE2, *fnGSFTot;
TH1D *g_hJIntrins;
TH2D *g_h2JIntrins;       // 2D histogram of underlying density
TF2 *fJIntrins;         // function to retreive underlying density

void InitFn() {
  fnLDa    = new TF1("fnLDa",    "GetLDa(x)",0,10);
  fnSpCut  = new TF1("fnSpCut",  "sqrt(GetSpinCut2(x))",0,10);
  fnGSFE1  = new TF1("fnGSFE1",  "GetStrE1([0],x)/x**3",0,20);
  fnGSFM1  = new TF1("fnGSFM1",  "GetStrM1(x)/x**3",0,20);
  fnGSFE2  = new TF1("fnGSFE2",  "GetStrE2(x)/x**5",0,20);
  fnGSFTot = new TF1("fnGSFTot", "GetStrE1([0],x)/x**3 + GetStrM1(x)/x**3 + GetStrE2(x)/x**5",0,20);

  double dEx = 0.5 * g_dExIMax; // spincut is slowly varying fn of E
  g_hJIntrins = new TH1D("hJIntrins","Underlying J Dist",
    int(g_dPlotSpMax), 0.0, int(g_dPlotSpMax) );
  double dSpinCut2 = GetSpinCut2(dEx);
  for(int spb=0; spb<int(g_dPlotSpMax); spb++) {
    double dSp;
    if(g_bIsEvenA){ dSp = spb; } else { dSp = spb + 0.5; }
    double dSpDen = (dSp + 0.5) * exp(-pow(dSp + 0.5, 2) / (2 * dSpinCut2))
       / dSpinCut2;
    g_hJIntrins->Fill(dSp,dSpDen);
  } // spb


  // Histogram over the Underlying J Dist in the continuum
  double dspStart=0; // startSpin to plot
  if(g_bIsEvenA){ dspStart=0; } else { dspStart+=0.5; }
  g_h2JIntrins = new TH2D("h2JIntrins","Underlying J Dist for E>E_crit",
                          g_dPlotSpMax+1,dspStart,int(g_dPlotSpMax+0.6+1),
                          g_nConEBin, g_dECrit, g_dExIMax);

  double adJInt[int(g_dPlotSpMax+0.6+1)]; // densities for the underlying J
  double adJIntNorm;                    // Normalization
  double dSp; // integer or half integer determination for density
  for(int ex=0; ex<g_nConEBin; ex++) {
    adJIntNorm = 0;
    // Get density adJInt for each spin
    for(int spb=0; spb<=g_dPlotSpMax; spb++) {
        if(g_bIsEvenA) dSp = spb; else dSp = spb + 0.5;
        adJInt[spb] = GetDensity(g_adConExCen[ex],dSp,0)+GetDensity(g_adConExCen[ex],dSp,1); // add both parities
        adJIntNorm += adJInt[spb];
    }
    // Normalize for each Ex bin
    for(int spb=0; spb<=g_dPlotSpMax; spb++) {
        if(g_bIsEvenA) dSp = spb; else dSp = spb + 0.5;
        adJInt[spb] /= adJIntNorm;
        g_h2JIntrins->Fill(dSp,g_adConExCen[ex],adJInt[spb]);
    }
  }
  // The distribution has no statistical errors, so need to set to 0
  for(int nbin=0; nbin<g_h2JIntrins->GetSize()+1; nbin++){
    g_h2JIntrins->SetBinError(nbin,0);
  }

// save continuum nld to file
TString smyFile = "NLDcont.dat";
ofstream ofNLD;
ofNLD.open(smyFile.Data());
ofNLD << "#Energy \t NLD" << endl;
for(int ex=0; ex<g_nConEBin; ex++) {
  double energy = g_adConExCen[ex];
  double rho = 0;
  // Get density adJInt for each spin
  for(int spb=0; spb<=g_nConSpbMax; spb++) {
      if(g_bIsEvenA) dSp = spb; else dSp = spb + 0.5;
      rho += GetDensity(g_adConExCen[ex],dSp,0)+GetDensity(g_adConExCen[ex],dSp,1); // add both parities
  }
  ofNLD << energy << "\t" << rho << endl;
}


} // InitFn

// Get mean and standard deviation of values in a vector v
// here, var^2 0 = (x0^2-x^2)/n, so without "bessel correction"
vector<double> GetMeanAndStdev(vector<double> &v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();
  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size());
  vector<double> meanAndStdev {mean, stdev};
  return meanAndStdev;
}

// Get average total radiative width
double GetGg(double dExcs, double dSpcs, int nParcs, int nReal, int nBins=1) {
  // initalize arrays to pass to GetWidth
  double *adDisWid;
  adDisWid = new double[g_nDisLvlMax]; // width to each discrete lvl
  double *adConWid; // width to each EJP bin (summed over in-bin lvls)
  adConWid   = new double  [g_nConEBin * g_nConSpbMax * 2](); // 0 init
  TRandom2 *arConState; // TRandom2 state for randoms
  arConState = new TRandom2[g_nConEBin * g_nConSpbMax * 2];

  int nSpcs = int(dSpcs);
  int nExBin = GetContExBin(dExcs);
  int nCSsum = 0;
  double dGg = 0;
  for (int nExb=nExBin-nBins/2;nExb<nExBin+(nBins-1)/2+1;nExb++){
    nCSsum += g_anConLvl[EJP(nExb,nSpcs,nParcs)]; // + #continuum states in a bin
  }
  vector<double> vGg;
  vGg.reserve(nCSsum);

  for (int nExb=nExBin-nBins/2;nExb<nExBin+(nBins-1)/2+1;nExb++){
    int nCS = g_anConLvl[EJP(nExb,nSpcs,nParcs)]; // #continuum states in a bin
    for(int nlvl=0; nlvl<nCS; nlvl++) { // find Gg of each state in the EJpi bin
      dGg = GetWidth(nExBin,nSpcs,nParcs,nlvl,nReal,
                              adConWid,adDisWid,arConState)*1e9;
      vGg.push_back(dGg);
      //cout << "Total width of a c.s. level " << nlvl << " is "<< dGg << " meV" << endl;
    }
  }

  vector<double> vGgMeanAndStdev = GetMeanAndStdev(vGg);
  double dGgMean  = vGgMeanAndStdev[0];
  double dGgStdev = vGgMeanAndStdev[1];

  cout<< "Number of states "<< nCSsum << " from " << nBins << " continuum bins; S_n +- Ex: " << nBins*g_dConESpac << endl;
  cout << "Average total width of the c.s. is " << dGgMean<< " meV" << " +- " << dGgStdev << endl;

  return dGgMean;
}

double GetMyGg(double dExcs=6.543, double dSpcs=0, int nParcs=0, int nReal=0, int nBins=1, int MaxBins=20) {
  // initalize arrays to pass to GetWidth
  double *adDisWid;
  adDisWid = new double[g_nDisLvlMax]; // width to each discrete lvl
  double *adConWid; // width to each EJP bin (summed over in-bin lvls)
  adConWid   = new double  [g_nConEBin * g_nConSpbMax * 2](); // 0 init
  TRandom2 *arConState; // TRandom2 state for randoms
  arConState = new TRandom2[g_nConEBin * g_nConSpbMax * 2];

  int nSpcs = int(dSpcs);
  int nExBin = GetContExBin(dExcs);
  int nCSsum = 0;
  double dGg = 0;
  for (int nExb=nExBin-nBins/2;nExb<nExBin+(nBins-1)/2+1;nExb++){
    nCSsum += g_anConLvl[EJP(nExb,nSpcs,nParcs)]; // + #continuum states in a bin
  }
  vector<double> vGg;
  vGg.reserve(nCSsum);

  for (int nExb=nExBin-nBins/2;nExb<nExBin+(nBins-1)/2+1;nExb++){
    int nCS = g_anConLvl[EJP(nExb,nSpcs,nParcs)]; // #continuum states in a bin
    for(int nlvl=0; nlvl<nCS; nlvl++) { // find Gg of each state in the EJpi bin
      dGg = GetWidth(nExBin,nSpcs,nParcs,nlvl,nReal,
                              adConWid,adDisWid,arConState)*1e9;
      vGg.push_back(dGg);
      if(nlvl==MaxBins){
        break;
      }
      //cout << "Total width of a c.s. level " << nlvl << " is "<< dGg << " meV" << endl;
    }
  }

  vector<double> vGgMeanAndStdev = GetMeanAndStdev(vGg);
  double dGgMean  = vGgMeanAndStdev[0];
  double dGgStdev = vGgMeanAndStdev[1];

  // cout<< "Number of states "<< nCSsum << " from " << nBins << " continuum bins; S_n +- Ex: " << nBins*g_dConESpac << endl;
  cout << "Average total width of the c.s. is " << dGgMean<< " meV" << " +- " << dGgStdev << endl;

  return dGgMean;
}


// Get average neutron resonance spacing D0 for s-wave neutorn capture
// note: For now this will be for the latest realization
double GetD0(double dEx, double dSp, int nPar, int nBins=1) {
  // dEx is the excitation energy (should be @Sn)
  // dSp is the target spin
  // nPar is the target parity
  // nBins is #Ex-bins to calc D0 from

  double rhoH = GetDensity(dEx, dSp+0.5, nPar);
  double rhoL = GetDensity(dEx, dSp-0.5, nPar);
  double D0_th_exact = 1./ (rhoH+rhoL)*1e6; // units of eV

  double rho_th = 0;
  double rho_real = 0;
  int nLevelsSum = 0;

  int nExBin = GetContExBin(dEx);
  for (int nExb=nExBin-nBins/2;nExb<nExBin+(nBins-1)/2+1;nExb++){
    dEx = g_adConExCen[nExb];

    rhoH = GetDensity(dEx, dSp+0.5, nPar);
    rhoL = GetDensity(dEx, dSp-0.5, nPar);
    rho_th += rhoH + rhoL;

    rho_real += g_anConLvl[EJP(GetContExBin(dEx),dSp+0.5,nPar)] / g_dConESpac;
    nLevelsSum += g_anConLvl[EJP(GetContExBin(dEx),dSp+0.5,nPar)];
    if(dSp>0){
      rho_real += g_anConLvl[EJP(GetContExBin(dEx),dSp-0.5,nPar)] / g_dConESpac;
      nLevelsSum += g_anConLvl[EJP(GetContExBin(dEx),dSp-0.5,nPar)];
    }
  }
  double D0_th = 1./(rho_th/nBins)*1e6; // units of eV
  double D0_real= 1./(rho_real/nBins) *1e6; // units of eV

  cout<<" s-wave neutron-capture resonance spacing, D0, from model, Ex exact: "<< D0_th_exact <<" eV"<<endl;
  cout<<" s-wave neutron-capture resonance spacing, D0, from model: "<< D0_th <<" eV"<<endl;
  cout<<" s-wave neutron-capture resonance spacing, D0, form realiation: "<< D0_real <<" eV"<<endl;
  cout<< "#States in bin(s): " << nLevelsSum << " from " << nBins << " continuum bins; Ex +- dEx: " << nBins*g_dConESpac << endl;;

  return D0_real;
}



TH2D *g_ah2PopLvl  [g_nReal][g_nExIMean];
TH1D *g_ahDisPop   [g_nReal][g_nExIMean];
TH1D *g_ahJPop     [g_nReal][g_nExIMean];
TH1D *g_ahTSC      [g_nReal][g_nExIMean][g_nDisLvlMax];
TH1D *g_ahDRTSC    [g_nReal][g_nExIMean][g_nDRTSC];
TH2D *g_ah2FeedTime[g_nReal][g_nExIMean];
TH2D *g_ah2ExEg    [g_nReal][g_nExIMean];
TH2D *g_ah21Gen    [g_nReal][g_nExIMean];
TH1D *g_ahGSpec    [g_nReal][g_nExIMean];
TH1D *g_ahICSpec   [g_nReal][g_nExIMean];
TH2D *g_ah2PopI    [g_nReal][g_nExIMean];
TGraph *g_grTotWidAvg       [g_nExIMean];


#ifdef bSaveTree
// to save data to trees
vector<Double_t> v_dEgs_save; // emitted gamma rays
double dExI_save; // initial excitation energy
int nJI_save; // initial spin (for uneven A: J=J-0.5)
int nPar_save; // initial parity (0 or 1: see elsewhere)
int real_save; // realization
vector<Double_t> v_dTimeToLvls_save; // decay time until this level
#endif // bSaveTree

/******************************************************************************/
/**************************** MAIN LOOP ***************************************/
/******************************************************************************/
void RAINIER(int g_nRunNum = 1) {
  cout << "Starting RAINIER" << endl;
  TTimeStamp tBegin;
  try {g_sRAINIERPath = string(std::getenv("RAINIER_PATH"));}
      catch (std::logic_error&) {
      	cout<< "RAINIER_PATH not set as environment variable"<< endl;
      	g_sRAINIERPath=".";};
  TString sSaveFile = TString::Format("Run%04d.root",g_nRunNum);
  TFile *fSaveFile = new TFile(sSaveFile, "recreate");
  ReadDisInputFile();
  #ifdef bPrintLvl
  PrintDisLvl();
  #endif
  #ifdef bExFullRxn
  ReadPopFile();
  #endif
  #ifdef bUseICC
  InitICC();
  #endif

  for(int exim=0; exim<g_nExIMean; exim++) {
    g_grTotWidAvg[exim] = new TGraph(g_nReal); // bench
  }

  #ifdef bSaveTree
  TTree *tree = new TTree("tree","RAINIER cascades tree");
  tree->Branch("Egs",&v_dEgs_save);
  tree->Branch("ExI",&dExI_save, "dExI_save/D");
  tree->Branch("JI_int",&nJI_save, "nJI_save/I");
  tree->Branch("nPar",&nPar_save, "nPar_save/I");
  tree->Branch("dTimeToLvls",&v_dTimeToLvls_save);
  #endif // bSaveTree

  ///////// Realization Loop /////////
  for(int real=0; real<g_nReal; real++) {
    BuildConstructed(real);
    #ifdef bPrintLvl
    PrintConLvl();
    #endif
    cout << "Realization " << real << endl;

    #ifdef bSaveTree
    if(g_nReal>1){
      tree->Branch("realiation",&real_save,"real_save/I");
    }
    #endif // bSaveTree

    ///////// Initial Excitation loop /////////
    for(int exim=0; exim<g_nExIMean; exim++) {
      double dExIMean = g_adExIMean[exim];
      double dExISpread = g_dExISpread; // could make resolution dep on ExIMean
      #ifdef bExSpread
      cout << "  Initial Excitation Mean: " << dExIMean << " +- "
        << dExISpread << " MeV" << endl;
      #endif

      ///// Initialize Histograms /////
      g_ah2PopLvl[real][exim] = new TH2D(
        Form("h2ExI%dPopLvl_%d",exim,real),
        Form("Population of Levels: %2.1f MeV, Real%d",dExIMean,real),
        2*g_dPlotSpMax, -g_dPlotSpMax, g_dPlotSpMax,
        g_dExIMax / g_dConESpac, 0, g_dExIMax);

      double dFeedTimeMax = (270-20) / (5.5-11.0) * dExIMean + 520; // fs
        // harder to pick out multistep decay at short times and low ExI
      int nFeedTimeBin = 300;
      g_ah2FeedTime[real][exim] = new TH2D(
        Form("h2ExI%dFeedTime_%d",exim,real),
        Form("Feeding Levels: %2.1f MeV, Real%d",dExIMean,real),
        g_nDisLvlMax,0,g_nDisLvlMax, nFeedTimeBin,0.0,dFeedTimeMax);

      int nBinEx = 300, nBinEg = 300; // mama, rhosigchi, etc. purposes
      g_ah2ExEg[real][exim] = new TH2D(
        Form("h2ExI%dEg_%d",exim,real),
        Form("E_{x,i} = %2.1f MeV vs. E_{#gamma}, Real%d",dExIMean,real),
        nBinEg,0,g_dExIMax*1000, nBinEx,0,g_dExIMax*1000);

      g_ah21Gen[real][exim] = new TH2D(
        Form("h2ExI%d1Gen_%d",exim,real),
        Form("E_{x,i} = %2.1f MeV 1st Generation, Real%d",dExIMean,real),
        nBinEg,0,g_dExIMax*1000, nBinEx,0,g_dExIMax*1000);

      g_ah2PopI[real][exim] = new TH2D(
        Form("h2ExI%dPopI_%d",exim,real),
        Form("E_{x,i} = %2.1f Events Populated in Real%d",dExIMean,real),
        g_dPlotSpMax,0,g_dPlotSpMax, 900,0,g_dExIMax);

      double dEgMax = g_dExIMax;
      for(int dis=0; dis<g_nDisLvlMax; dis++) {
        double dDisEne = g_adDisEne[dis];
        double dDisSp  = g_adDisSp [dis];
        int nDisPar    = g_anDisPar[dis];
        if(nDisPar == 0) // negative parity
          g_ahTSC[real][exim][dis] = new TH1D(
            Form("hExI%dto%dTSC_%d",exim,dis,real),
            Form("TSC to lvl %2.1f- %2.3f MeV, Real%d",
            dDisSp,dDisEne,real),
            g_nEgBin,0.0,dEgMax);
        else // positive parity
          g_ahTSC[real][exim][dis] = new TH1D(
            Form("hExI%dto%dTSC_%d",exim,dis,real),
            Form("TSC to lvl %2.1f+ %2.3f MeV, Real%d",
            dDisSp,dDisEne,real),
            g_nEgBin,0.0,dEgMax);
      } // TSC to discrete lvl

      for(int prim2=0; prim2<g_nDRTSC; prim2++) {
        // dont want to make into th2d and do projections later
        // like I did with feedAnalysis, was too time coding time costly
        g_ahDRTSC[real][exim][prim2] = new TH1D(
          Form("hExI%dDRTSC_%d_%d",exim,prim2,real),
          Form("Primary 2^{+}: %2.1f MeV, Real%d",dExIMean,real),
          g_nEgBin,0.0,dEgMax);
      } // prim2

      g_ahGSpec[real][exim] = new TH1D(
        Form("hExI%dGSpec_%d",exim,real),
        Form("Gamma Spectrum: %2.1f MeV, Real%d",dExIMean,real),
        g_nEgBin,0.00,dEgMax);

      g_ahICSpec[real][exim] = new TH1D(
        Form("hExI%dICSpec_%d",exim,real),
        Form("Internal Conv Spectrum: %2.1f MeV, Real%d",dExIMean,real),
        g_nEgBin,0.00,dEgMax);

      g_ahDisPop[real][exim] = new TH1D(
        Form("hExI%dDisPop_%d",exim,real),
        Form("Discrete Populations: %2.1f MeV, Real%d",dExIMean,real),
        g_nDisLvlMax,0,g_nDisLvlMax);

      g_ahJPop[real][exim] = new TH1D(
        Form("hExI%dJPop_%d",exim,real),
        Form("Spin Initial Pop: %2.1f MeV, Real%d",dExIMean,real),
        int(g_dPlotSpMax),0,g_dPlotSpMax); // wont have this plot for half int J

      #ifdef bExSingle
      // save initial widths and rands so dont need to recompute
      double *adDisWid1     = new double  [g_nDisLvlMax];
      double *adConWid1     = new double  [g_nConEBin * g_nConSpbMax * 2]();
      TRandom2 *arConState1 = new TRandom2[g_nConEBin * g_nConSpbMax * 2];

      TRandom2 ranEv1(1); // unused
      int nExI1,nSpbI1,nParI1,nDisEx1,nLvlInBinI1;
      GetExI(nExI1, nSpbI1, nParI1, nDisEx1, nLvlInBinI1, ranEv1,
        1.0, 1.0); // unused
      int nConEx1 = nExI1, nSpb1 = nSpbI1, nPar1 = nParI1,
        nLvlInBin1 = nLvlInBinI1;
      cout << "Getting 1st step widths" << endl;
      double dTotWid1 = GetWidth(
        nConEx1, nSpb1, nPar1, nLvlInBin1, real,
        adConWid1, adDisWid1, arConState1);
      cout << "Starting decay events" << endl;
      #endif

      #ifdef bParallel
      #pragma omp parallel// if resize cmd window while running - will stall
      #endif // parallel
      { // each active processor gets an allocated array
        // dont have to renew with each event
        //double adDisWid[g_nDisLvlMax]; // width to each discrete level
        double *adDisWid;
        adDisWid = new double[g_nDisLvlMax]; // wid to each discrete lvl
        double *adConWid; // width to each EJP bin (summed over in-bin lvls)
        adConWid   = new double  [g_nConEBin * g_nConSpbMax * 2](); // 0 init
        TRandom2 *arConState; // TRandom2 state for randoms
        arConState = new TRandom2[g_nConEBin * g_nConSpbMax * 2];

        int nEle = 0;
        #ifdef bParallel
        int nCount = 0;
        int nThreads = omp_get_num_threads();
        #pragma omp for
        #endif
        ////////////////////////////////////////////////////////////////////////
        /////// EVENT LOOP /////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        for(int ev=0; ev<g_nEvent; ev++) {
          #ifdef bParallel
          nCount++;
          if( !( (nCount * nThreads) % g_nEvUpdate) )
            fprintf(stderr,"    approx: %d / %d \r",
              nCount * nThreads, g_nEvent);
          #else
          if( !(ev % g_nEvUpdate) )
            cout << "    " << ev << " / " << g_nEvent << "\r" << flush;
          #endif
          TRandom2 ranEv(1 + real + ev * g_nReal);

          /////// initial state formation ///////
          int nExI,nSpbI,nParI,nDisEx,nLvlInBinI; // Initial state variables
          GetExI(nExI, nSpbI, nParI, nDisEx, nLvlInBinI, ranEv, // init vars
            dExIMean, dExISpread); // with these inputs from experiment
          int nConEx = nExI, nSpb = nSpbI, nPar = nParI,
            nLvlInBin = nLvlInBinI; // set initial to working state variables
          g_ahJPop[real][exim]->Fill(nSpbI);

          double dExI;
          if(nConEx < 0) { // in discrete
            dExI = g_adDisEne[nDisEx];
          } else { // in constructed scheme
            dExI = GetInBinE(real,nExI,nSpbI,nParI,nLvlInBinI);
          }
          g_ah2PopI[real][exim]->Fill(nSpbI,dExI);

          double dTimeToLvl = 0.0;
          int nStep = 0;
          bool bIsAlive = true;
          double dEg1 = 0.0, dEg2 = 0.0; // steps for TSC
          bool bHadEle = false;
          #ifdef bSaveTree
          vector<Double_t> v_dEgs;
          vector<Double_t> v_dTimeToLvls;
          #endif // bSaveTree
          /////// Decay till ground state ////////
          while(nDisEx > 0 && bIsAlive) { // nDisEx > g.s. (i.e. excited)
            // bIsAlive: could get stuck in constructed high spin state and have
              // no dipole or quadrupole decay option: effectively an isomer

            ///// pre decay /////
            double dExPre, dMixDelta2;
            if(nConEx < 0) { // in discrete
              dExPre  = g_adDisEne[nDisEx];
            } else { // in constructed scheme
              dExPre  = GetInBinE(real,nConEx,nSpb,nPar,nLvlInBin);
            } // pre decay

            // Fill populations
            if(nPar == 1) g_ah2PopLvl[real][exim]->Fill(nSpb + 0.5, dExPre);
            else g_ah2PopLvl[real][exim]->Fill(-nSpb - 0.5, dExPre);
              // +-0.5 so can plot 0+ and 0-

            ///// during decay /////
            int nTransMade = 0; // want to know multipole and character for ICC
            if(nConEx < 0) { ///// in discrete /////
              double dTotWid = 0.0;
              double dLifeT = g_adDisT12[nDisEx] / log(2); // lifetime from file
              double dDecayTime = ranEv.Exp(dLifeT);
              dTimeToLvl += dDecayTime;
              bIsAlive = TakeStep( // no variable change if !bIsAlive
                nConEx, nSpb, nPar, nDisEx, nLvlInBin,
                nTransMade, dMixDelta2, dTotWid, real,
                adConWid, adDisWid, arConState, ranEv);
            } else { ///// in constructed scheme //////
              #ifdef bExSingle
              double dTotWid;
              if(nConEx == g_nConEBin - 1) { // use saved init width
                dTotWid = dTotWid1;
              } else { // not at intial state
                dTotWid = GetWidth(
                  nConEx, nSpb, nPar, nLvlInBin, real,
                  adConWid, adDisWid, arConState);
              } // Ex single
              #else
              double dTotWid = GetWidth(
                nConEx, nSpb, nPar, nLvlInBin, real,
                adConWid, adDisWid, arConState);
              #endif
              if(ev == 0 && nStep == 0) { // bench
                if(real != 0) {
                  double dOldAvg, dReal;
                  g_grTotWidAvg[exim]->GetPoint(real-1, dReal, dOldAvg);
                  double dNewAvg = (dOldAvg * real + dTotWid) / double(real+1);
                  g_grTotWidAvg[exim]->SetPoint(real, real, dNewAvg);
                } else { // 1st real
                  g_grTotWidAvg[exim]->SetPoint(real, real, dTotWid);
                }
              } // bench
              #ifdef bExSingle
              if(nConEx == g_nConEBin - 1) { // use saved init widths and rands
                dTimeToLvl += GetDecayTime(dTotWid1, ranEv);
                bIsAlive = TakeStep(
                  nConEx, nSpb, nPar, nDisEx, nLvlInBin,
                  nTransMade, dMixDelta2, dTotWid1, real,
                  adConWid1, adDisWid1, arConState1, ranEv);
              } else { // not at initial state
                dTimeToLvl += GetDecayTime(dTotWid, ranEv);
                bIsAlive = TakeStep(
                  nConEx, nSpb, nPar, nDisEx, nLvlInBin,
                  nTransMade, dMixDelta2, dTotWid, real,
                  adConWid, adDisWid, arConState, ranEv);
              } // Ex Single
              #else
              dTimeToLvl += GetDecayTime(dTotWid, ranEv);
              bIsAlive = TakeStep(
                nConEx, nSpb, nPar, nDisEx, nLvlInBin,
                nTransMade, dMixDelta2, dTotWid, real,
                adConWid, adDisWid, arConState, ranEv);
              #endif
            } // end of decay
            nStep++;

            ///// post decay /////
            if(bIsAlive) { // not stuck in isomeric state: emission ignored
              double dExPost;
              if(nConEx < 0) { // in discrete
                dExPost = g_adDisEne[nDisEx];
                // Fill low lying feeding time
                g_ah2FeedTime[real][exim]->Fill(nDisEx, dTimeToLvl);
                g_ahDisPop[real][exim]->Fill(nDisEx);
              } else { // in constructed scheme
                dExPost = GetInBinE(real,nConEx,nSpb,nPar,nLvlInBin);
              }

              double dEg = dExPre - dExPost;
              #ifdef bSaveTree
              v_dEgs.push_back(dEg);
              v_dTimeToLvls.push_back(dTimeToLvl);
              #endif // bSaveTree
              ///// Internal Conversion /////
              double dICC = GetICC(dEg,nTransMade,dMixDelta2);
              double dProbEle = dICC / (1.0 + dICC);
              double dRanICC = ranEv.Uniform(1.0);
              bool bIsElectron = false;
              if(dProbEle > dRanICC) bIsElectron = true;

              if( !bIsElectron ) { // emitted gamma
                double dExRes = g_dExRes; // ~ particle resolution
                double dExDet = dExI + ranEv.Gaus(0.0, dExRes);
                g_ahGSpec[real][exim]->Fill(dEg);
                g_ah2ExEg[real][exim]->Fill(dEg*1000, dExDet*1000);
                if(nStep == 1)
                  g_ah21Gen[real][exim]->Fill(dEg*1000, dExDet*1000);

                // TSC spectra
                if(nStep == 1) dEg1 = dEg;
                if(nStep == 2) dEg2 = dEg;
                // DRTSC spectra
                if(nStep == 1 && nDisEx < g_nDisLvlMax) { // 1st step discrete
                  for(int prim2=0; prim2<g_nDRTSC; prim2++) {
                    if(g_anDRTSC[prim2] == nDisEx) { // primary is known 2+
                      g_ahDRTSC[real][exim][prim2]->Fill(dEg); // neglect ICC
                    } // primary is known 2+
                  } // check if primary
                } // 1st step to discrete
              } else { // was electron
                // if you want an IC spectrum, need to separate out individual
                // XL components from BrIcc. Also need to read electronic shell
                // energies then subtract them from the transition E:
                // g_ahICSpec[real][exim]->Fill(dEg - dElecBindEne);
                bHadEle = true; // at least one electron ruins TSC
                nEle++;
              } // ICC check

              // TSC spectra to specific states
              if(nStep == 2 && !bHadEle) {
                for(int dis=0; dis<g_nDisLvlMax; dis++) {
                  if(nDisEx == dis) {
                    g_ahTSC[real][exim][dis]->Fill(dEg1);
                    g_ahTSC[real][exim][dis]->Fill(dEg2);
                  } // discrete match
                } // end on discrete
              } // 2 steps

            } // IsAlive

          } // no longer excited

          // saving to tree; slightly hacky coding in order to
          // ensure that it is thread save, without creating
          // one tree per thread
          #ifdef bSaveTree
            #ifdef bParallel
            #pragma omp critical
            {
            #endif //bParallel
            v_dEgs_save = v_dEgs;
            v_dTimeToLvls_save = v_dTimeToLvls;
            dExI_save = dExI;
            nJI_save = nSpbI;
            nPar_save = nParI;
            real_save = real;
            tree->Fill();
            #ifdef bParallel
            } // critical
            #endif
          #endif // bSaveTree

          // Save progress every g_nEvSave events
          if( !(ev % g_nEvSave) ) {
          // only one thread at a time: prevent file corruption
            #ifdef bParallel
            #pragma omp critical
            #endif
              {
              cout << "    " << ev << " / " << g_nEvent << "\r" << flush;
              cout << "Saving Progress" << endl;
              fSaveFile->Write("",TObject::kOverwrite);
              } // omp critical
            } // g_nEvSave
        } //////////////////////////////////////////////////////////////////////
        ////////////////////////// EVENTS //////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        // deallocate memory
        delete[] adConWid;
        delete[] arConState;
        adConWid   = 0;
        arConState = 0;
        #ifdef bExSingle
        delete[] adConWid;
        delete[] arConState;
        adConWid   = 0;
        arConState = 0;
        #endif
        //cout << "    " << nEle << " internal conversions" << endl;
      } // parallel
      cout << endl << "    " << g_nEvent << " / " << g_nEvent << " processed"
        << endl << endl;

      ///// plotting preferences /////
      TH2D *g_ah2Temp[] = {
        g_ah2PopLvl  [real][exim],
        g_ah2FeedTime[real][exim],
        g_ah2ExEg    [real][exim],
        g_ah21Gen    [real][exim],
        g_ah2PopI    [real][exim]
        };
      int nh2Temp = 5;
      for(int h=0; h<nh2Temp; h++) {
        g_ah2Temp[h]->GetXaxis()->SetTitleSize(0.055);
        g_ah2Temp[h]->GetXaxis()->SetTitleFont(132);
        g_ah2Temp[h]->GetXaxis()->SetTitleOffset(0.8);
        g_ah2Temp[h]->GetXaxis()->CenterTitle();

        g_ah2Temp[h]->GetYaxis()->SetTitleSize(0.055);
        g_ah2Temp[h]->GetYaxis()->SetTitleFont(132);
        g_ah2Temp[h]->GetYaxis()->SetTitleOffset(0.65);
        g_ah2Temp[h]->GetYaxis()->CenterTitle();

        g_ah2Temp[h]->GetZaxis()->SetTitleSize(0.045);
        g_ah2Temp[h]->GetZaxis()->SetTitleFont(132);
        g_ah2Temp[h]->GetZaxis()->SetTitleOffset(-0.4);
        g_ah2Temp[h]->GetZaxis()->CenterTitle();
        g_ah2Temp[h]->GetZaxis()->SetTitle("Counts");
      } // hists
      g_ah2PopLvl  [real][exim]->GetXaxis()->SetTitle("J#Pi");
      g_ah2PopLvl  [real][exim]->GetXaxis()->SetNdivisions(20,0,0,kFALSE);
      g_ah2PopLvl  [real][exim]->GetYaxis()->SetTitle("E_{x} (MeV)");
      g_ah2FeedTime[real][exim]->GetXaxis()->SetTitle("Discrete Level Number");
      g_ah2FeedTime[real][exim]->GetYaxis()->SetTitle("Feeding Time (fs)");
      g_ah2ExEg    [real][exim]->GetXaxis()->SetTitle("E_{#gamma} (MeV)");
      g_ah2ExEg    [real][exim]->GetYaxis()->SetTitle("E_{x,I} (MeV)");
      g_ah2PopI    [real][exim]->GetXaxis()->SetTitle("J_{I} (#hbar)");
      g_ah2PopI    [real][exim]->GetYaxis()->SetTitle("E_{x,I} (MeV)");

      // TH1D:
      TH1D *g_ah1Temp[] = {
        g_ahGSpec [real][exim],
        //g_ahTSC   [real][exim],
        g_ahDisPop[real][exim],
        g_ahJPop  [real][exim],
        g_ahDRTSC [real][exim][0] // 1st determines axes in plot
      };
      int nh1Temp = 4;
      for(int h=0; h<nh1Temp; h++) {
        g_ah1Temp[h]->GetXaxis()->SetTitleSize(0.055);
        g_ah1Temp[h]->GetXaxis()->SetTitleFont(132);
        g_ah1Temp[h]->GetXaxis()->SetTitleOffset(0.8);
        g_ah1Temp[h]->GetXaxis()->CenterTitle();
        g_ah1Temp[h]->GetXaxis()->SetTitle("E_{#gamma} (MeV)");

        g_ah1Temp[h]->GetYaxis()->SetTitleSize(0.055);
        g_ah1Temp[h]->GetYaxis()->SetTitleFont(132);
        g_ah1Temp[h]->GetYaxis()->SetTitleOffset(0.85);
        g_ah1Temp[h]->GetYaxis()->CenterTitle();
        g_ah1Temp[h]->GetYaxis()->SetTitle("Counts");
      } // hists

    } // Excitation mean
  } // realization
  InitFn();


  cout << "Writing Histograms" << endl;
  fSaveFile->Write("",TObject::kOverwrite); // only saves histograms, not the parameters, nor TF1s
  ////// save parameters /////
  #define SAVE_PAR(stream,variable) (stream) <<#variable" "<<(variable) << endl
  #define SAVE_ARR(stream,variable,size) (stream) <<#variable<<" "; for(int i=0; i<size; i++) { (stream) << (variable[i]) << " "; }; (stream) << endl;
  TString sParFile = TString::Format("Param%04d.dat", g_nRunNum);
  ofstream ofParam;
  ofParam.open(sParFile.Data());
  SAVE_PAR(ofParam,g_dECrit);
  SAVE_PAR(ofParam,g_dExISpread);
  SAVE_PAR(ofParam,g_dExIMax);
  SAVE_PAR(ofParam,g_dPlotSpMax);
  SAVE_PAR(ofParam,g_nReal);
  SAVE_PAR(ofParam,g_nExIMean);
  SAVE_PAR(ofParam,g_nEvent);
  SAVE_PAR(ofParam,g_nDRTSC);
  SAVE_PAR(ofParam,g_bIsEvenA);
  SAVE_PAR(ofParam,g_nDRTSC);
  SAVE_ARR(ofParam,g_anDRTSC,g_nDRTSC);
  SAVE_PAR(ofParam,g_nPopLvl);
  SAVE_ARR(ofParam,g_anPopLvl,g_nPopLvl);
  SAVE_PAR(ofParam,g_nDisLvlMax);
  SAVE_ARR(ofParam,g_adDisSp,g_nDisLvlMax);
  SAVE_ARR(ofParam,g_anDisPar,g_nDisLvlMax);
  SAVE_ARR(ofParam,g_adDisEne,g_nDisLvlMax);
  SAVE_ARR(ofParam,g_adExIMean,g_nExIMean);

  TString sInputFile = TString::Format("Input%04d.dat", g_nRunNum);
  ofstream ofInput;
  ifstream ifInput;
  ifInput.open("RAINIER.C");
  ofInput.open(sInputFile.Data());
  string sLine;
  while( getline(ifInput,sLine) ) {
    ofInput << sLine << endl;
  }

  TTimeStamp tEnd;
  double dElapsedSec = double(tEnd.GetSec() - tBegin.GetSec());
  cout << "Time elapsed during RAINIER execution: " << dElapsedSec << " sec"
    << endl;
  gROOT->ProcessLine(".L $RAINIER_PATH/Analyze.C++"); // load the separate analysis file
  gROOT->ProcessLine("RetrievePars()"); // linking files is always wonky in ROOT
} // main

// copy of the function above to ensure that one can run RAINIER directly, or runRAINIER.sh in a seperate folder
void RAINIER_copy(int g_nRunNum = 1){
	RAINIER(g_nRunNum);
}
