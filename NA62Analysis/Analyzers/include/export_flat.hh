#ifndef EXPORT_FLAT_HH
#define EXPORT_FLAT_HH

#include <stdlib.h>
#include <vector>
#include "Analyzer.hh"
#include <TCanvas.h>
#include "SpectrometerTrackVertex.hh"

class TH1I;
class TH2F;
class TGraph;
class TTree;
class DownstreamTrack;
class EnergyCluster;

struct TrackStruct
{
  Bool_t fExists = false;
  Bool_t fHasMUV3 = false;
  Int_t fRICHMLH = -1;
  Int_t fRICHNHits = -1;
  Int_t fCharge = -99;
  Float_t fRICHRadius = -1;
  Float_t fRICHCenterX = -1;
  Float_t fRICHCenterY = -1;
  Float_t fDirectionX = 0.0;
  Float_t fDirectionY = 0.0;
  Float_t fDirectionZ = 0.0;
  Float_t fDirectionAfterMagnetX = 0.0;
  Float_t fDirectionAfterMagnetY = 0.0;
  Float_t fDirectionAfterMagnetZ = 0.0;
  Float_t fPosAfterMagnetX = 0.0;
  Float_t fPosAfterMagnetY = 0.0;
  Float_t fMomentumMag = 0.0;
  Float_t fTime = -999.0;
  Float_t fLKrEnergy = -99.0;
  Float_t fMUV3Time = -999.;

  void Reset() {
      fExists = false;
      fHasMUV3 = false;
      fRICHMLH = -1;
      fRICHNHits = -1;
      fCharge = -99;
      fRICHRadius = -1;
      fRICHCenterX = -1;
      fRICHCenterY = -1;
      fDirectionX = 0.0;
      fDirectionY = 0.0;
      fDirectionZ = 0.0;
      fDirectionAfterMagnetX = 0.0;
      fDirectionAfterMagnetY = 0.0;
      fDirectionAfterMagnetZ = 0.0;
      fPosAfterMagnetX = 0.0;
      fPosAfterMagnetY = 0.0;
      fMomentumMag = 0.0;
      fTime = -999.0;
      fLKrEnergy = -99.0;
      fMUV3Time = -999.;
  }
};
struct GammaStruct
{
  Bool_t fExists = false;
  Float_t fPositionX = 0.0;
  Float_t fPositionY = 0.0;
  Float_t fLKrEnergy = 0.0;
  Float_t fTime = -999.0;

  void Reset() {
    fExists = false;
    fPositionX = 0.0;
    fPositionY = 0.0;
    fLKrEnergy = 0.0;
    fTime = -999.0;
  }
};
struct BeamStruct
{
  Float_t fMomentum = 0.0;
  Float_t fDirectionX = 0.0;
  Float_t fDirectionY = 0.0;
  Float_t fDirectionZ = 0.0;
  Float_t fPosX = 0.0;
  Float_t fPosY = 0.0;
  Float_t fPosZ = 0.0;
};
struct VertexStruct
{
  Float_t fX = 0.0;
  Float_t fY = 0.0;
  Float_t fZ = 0.0;

  void Reset() {
    fX = 0.0;
    fY = 0.0;
    fZ = 0.0;
  }
};

struct FlatStruct
{
  BeamStruct fBeam;
  VertexStruct fVertex;
  TrackStruct track1;
  TrackStruct track2;
  TrackStruct track3;
  GammaStruct clus1;
  GammaStruct clus2;
};

class export_flat : public NA62Analysis::Analyzer
{
public:
  explicit export_flat(NA62Analysis::Core::BaseAnalysis *ba);
  ~export_flat();
  void InitHist();
  void InitOutput();
  void ProcessSpecialTriggerUser(int iEvent, unsigned int triggerType);
  void Process(int iEvent);
  void PostProcess();
  void StartOfBurstUser();
  void EndOfBurstUser();
  void StartOfRunUser();
  void EndOfRunUser();
  void EndOfJobUser();
  void DrawPlot();
protected:
  void BranchTrack(TrackStruct &ts, int it);
  void BranchCluster(GammaStruct &ts, int it);
  void FillTrack(DownstreamTrack &t, TrackStruct &ts, float refTime);
  void FillCluster(EnergyCluster &c, GammaStruct &ts);
  void Reset();

  Int_t bestInTimeVertices(std::vector<SpectrometerTrackVertex> &vtc);
  std::vector<int> goodDSTracks(std::vector<DownstreamTrack> &ds);
  Bool_t autopassSelection(std::vector<SpectrometerTrackVertex> &vertex3, std::vector<DownstreamTrack> &dsTracks, Int_t &goodvtx, Int_t &goodtrack);
  std::vector<int> additionalClusters(std::vector<EnergyCluster> &clusters);

  TTree *myTree;
  FlatStruct flat;
  Int_t fEventType; // 1 k3pi, 2 ke3, 3 kmu2, 4 k2pi, 5 kmu3, 6 bckg
  Int_t fRunNumber;
  Int_t fBurstNumber;

  Int_t fDownscaling;
  Int_t fDSCount;
  Double_t fReferenceTime;
};
#endif
