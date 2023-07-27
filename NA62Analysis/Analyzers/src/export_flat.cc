#include "export_flat.hh"

#include "Event.hh"
#include "Persistency.hh"
#include "functions.hh"
#include "DownstreamTrack.hh"
#include "SpectrometerTrackVertex.hh"
#include "BeamParameters.hh"
#include "Pi0Selection.hh"
#include "EnergyCluster.hh"
#include "TwoLinesCDA.hh"
#include "Kmu3SelectionNoSpectrometer.hh"

#include <iostream>
#include <stdlib.h>
#include <TH2I.h>
using namespace NA62Analysis;

export_flat::export_flat(Core::BaseAnalysis *ba): Analyzer(ba, "export_flat") {
  RequestTree(new TRecoLKrEvent);
  RequestTree(new TRecoSpectrometerEvent);
  RequestTree(new TRecoRICHEvent);
  RequestTree(new TRecoMUV3Event);
  RequestTree(new TRecoCedarEvent);
  RequestTree(new TRecoGigaTrackerEvent);

  fDSCount = 0;
  fDownscaling = 1000;
}

void export_flat::BranchTrack(TrackStruct &ts, int it){
  myTree->Branch(TString::Format("track%i_exists", it), &(ts.fExists));
  myTree->Branch(TString::Format("track%i_has_muv3", it), &(ts.fHasMUV3));
  myTree->Branch(TString::Format("track%i_rich_hypothesis", it), &(ts.fRICHMLH));
  myTree->Branch(TString::Format("track%i_rich_radius", it), &(ts.fRICHRadius));
  myTree->Branch(TString::Format("track%i_rich_nhits", it), &(ts.fRICHNHits));
  myTree->Branch(TString::Format("track%i_rich_center_x", it), &(ts.fRICHCenterX));
  myTree->Branch(TString::Format("track%i_rich_center_y", it), &(ts.fRICHCenterY));
  myTree->Branch(TString::Format("track%i_direction_x", it), &(ts.fDirectionX));
  myTree->Branch(TString::Format("track%i_direction_y", it), &(ts.fDirectionY));
  myTree->Branch(TString::Format("track%i_direction_z", it), &(ts.fDirectionZ));
  myTree->Branch(TString::Format("track%i_momentum_mag", it), &(ts.fMomentumMag));
  myTree->Branch(TString::Format("track%i_time", it), &(ts.fTime));
  myTree->Branch(TString::Format("track%i_lkr_energy", it), &(ts.fLKrEnergy));
  myTree->Branch(TString::Format("track%i_charge", it), &(ts.fCharge));
}

void export_flat::BranchCluster(GammaStruct &ts, int it){
  myTree->Branch(TString::Format("cluster%i_exists", it), &(ts.fExists));
  myTree->Branch(TString::Format("cluster%i_lkr_energy", it), &(ts.fLKrEnergy));
  myTree->Branch(TString::Format("cluster%i_position_x", it), &(ts.fPositionX));
  myTree->Branch(TString::Format("cluster%i_position_y", it), &(ts.fPositionY));
  myTree->Branch(TString::Format("cluster%i_time", it), &(ts.fTime));
}

void export_flat::InitOutput() {
  myTree = new TTree("NA62Flat", "NA62Flat");

  myTree->Branch("run", &fRunNumber);
  myTree->Branch("burst", &fBurstNumber);
  myTree->Branch("event_type", &fEventType);
  myTree->Branch("event_time", &fReferenceTime);
  myTree->Branch("beam_momentum_mag", &(flat.fBeam.fMomentum));
  myTree->Branch("beam_direction_x", &(flat.fBeam.fDirectionX));
  myTree->Branch("beam_direction_y", &(flat.fBeam.fDirectionY));
  myTree->Branch("beam_direction_z", &(flat.fBeam.fDirectionZ));
  myTree->Branch("beam_position_x", &(flat.fBeam.fPosX));
  myTree->Branch("beam_position_Y", &(flat.fBeam.fPosY));
  myTree->Branch("beam_position_z", &(flat.fBeam.fPosZ));
  myTree->Branch("vtx_x", &(flat.fVertex.fX));
  myTree->Branch("vtx_y", &(flat.fVertex.fY));
  myTree->Branch("vtx_z", &(flat.fVertex.fZ));
  BranchTrack(flat.track1, 1);
  BranchTrack(flat.track2, 2);
  BranchTrack(flat.track3, 3);
  BranchCluster(flat.clus1, 1);
  BranchCluster(flat.clus2, 2);
}

void export_flat::InitHist() {

  BookHisto(new TH1I("muv3tdiff", "muv3tdiff", 100, -10, 10));
  const char *selections[7]  = {"Events", "k3pi", "ke3", "kmu2", "k2pi", "kmu3", "autopass"};
  TH2I *h = new TH2I("sel_matrix", "sel_matrix", 6, -0.5, 5.5, 7, -0.5, 6.5);
  BookHisto(h);
  for (int i =0; i<7; ++i){
    h->GetXaxis()->SetBinLabel(i+1, selections[i]);
    h->GetYaxis()->SetBinLabel(i+1, selections[i]);
  }
}

void export_flat::StartOfRunUser() {
}

void export_flat::StartOfBurstUser() {
}

void export_flat::ProcessSpecialTriggerUser(int , unsigned int ) {
}

void export_flat::Process(int) {
  fRunNumber = GetRunID();
  fBurstNumber = GetBurstID();

  std::vector<DownstreamTrack> dsTracks = *GetOutput<std::vector<DownstreamTrack>>("DownstreamTrackBuilder.Output");
  std::vector<SpectrometerTrackVertex> vertices3 = *GetOutput<std::vector<SpectrometerTrackVertex>>("SpectrometerVertexBuilder.Output3");
  // std::vector<SpectrometerTrackVertex> vertices2 = *GetOutput<std::vector<SpectrometerTrackVertex>>("SpectrometerVertexBuilder.Output2");

  fReferenceTime = GetEventHeader()->GetFineTime() * TdcCalib;

  Bool_t k3pi_sel = *GetOutput<Bool_t>("K3piSelection.EventSelected");
  Int_t  k3piVertex = *GetOutput<Int_t>("K3piSelection.VertexID");

  Bool_t ke3_sel = *GetOutput<Bool_t>("Ke3Selection.EventSelected");
  Int_t  ke3Track = *GetOutput<Int_t>("Ke3Selection.Ke3TrackID");
  Pi0SelectionOutput ke3Pi0 = *GetOutput<Pi0SelectionOutput>("Ke3Selection.Ke3Pi0SelectionOutput");

  Bool_t kmu2_sel = *GetOutput<Bool_t>("Kmu2Selection.EventSelected");
  Int_t  kmu2Track = *GetOutput<Int_t>("Kmu2Selection.Kmu2TrackID");

  Bool_t kmu3_sel = *GetOutput<Bool_t>("Kmu3Selection.EventSelected");
  Int_t  kmu3Track = *GetOutput<Int_t>("Kmu3Selection.Kmu3TrackID");
  Pi0SelectionOutput kmu3Pi0 = *GetOutput<Pi0SelectionOutput>("Kmu3Selection.Kmu3Pi0SelectionOutput");

  Bool_t k2pi_sel = *GetOutput<Bool_t>("K2piSelection.EventSelected");
  Int_t  k2piTrack = *GetOutput<Int_t>("K2piSelection.K2piTrackID");
  Pi0SelectionOutput k2piPi0 = *GetOutput<Pi0SelectionOutput>("K2piSelection.K2piPi0SelectionOutput");
  TVector3 k2piVtx = *GetOutput<TVector3>("K2piSelection.K2piVertexPosition");

  Int_t autopassVtx = -1;
  Int_t autopassTrack = -1;
  Bool_t autopass_sel = autopassSelection(vertices3, dsTracks, autopassVtx, autopassTrack);

  FillHisto2("sel_matrix", "Events", "Events", 1);
  if(k3pi_sel){
    FillHisto2("sel_matrix", "k3pi", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "k3pi", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "k3pi", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "k3pi", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "k3pi", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "k3pi", "autopass", 1);
  }
  if(ke3_sel){
    FillHisto2("sel_matrix", "ke3", "ke3", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "ke3", "k3pi", 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "ke3", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "ke3", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "ke3", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "ke3", "autopass", 1);
  }
  if(kmu2_sel){
    FillHisto2("sel_matrix", "kmu2", "kmu2", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "kmu2", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "kmu2", "ke3" , 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "kmu2", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "kmu2", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "kmu2", "autopass", 1);
  }
  if(k2pi_sel){
    FillHisto2("sel_matrix", "k2pi", "k2pi", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "k2pi", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "k2pi", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "k2pi", "kmu2", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "k2pi", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "k2pi", "autopass", 1);
  }
  if(kmu3_sel){
    FillHisto2("sel_matrix", "kmu3", "kmu3", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "kmu3", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "kmu3", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "kmu3", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "kmu3", "k2pi", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "kmu3", "autopass", 1);
  }
  if(autopass_sel){
    FillHisto2("sel_matrix", "autopass", "autopass", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "autopass", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "autopass", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "autopass", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "autopass", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "autopass", "kmu3", 1);
  }
  Int_t nSel = k3pi_sel + ke3_sel + kmu2_sel + k2pi_sel + kmu3_sel;
  if((nSel!=1) & !autopass_sel) return; // Allow exactly one positive selection, or autopass

  flat.track1.fExists = false;
  flat.track2.fExists = false;
  flat.track3.fExists = false;
  flat.clus1.fExists = false;
  flat.clus2.fExists = false;

  BeamParameters *beamPar = BeamParameters::GetInstance();
  flat.fBeam.fMomentum = beamPar->GetBeamMomentum();
  TVector3 beam = beamPar->GetBeamThreeMomentum().Unit();
  flat.fBeam.fDirectionX = beam.X();
  flat.fBeam.fDirectionY = beam.Y();
  flat.fBeam.fDirectionZ = beam.Z();
  flat.fBeam.fPosX = beamPar->GetBeamX();
  flat.fBeam.fPosY = beamPar->GetBeamY();
  flat.fBeam.fPosZ = beamPar->GetBeamZ();

  if(autopass_sel) {
    fEventType = 5;

    if(autopassVtx!=-1){
      DownstreamTrack &t1 = dsTracks[vertices3[autopassVtx].GetTrackIndex(0)];
      DownstreamTrack &t2 = dsTracks[vertices3[autopassVtx].GetTrackIndex(1)];
      DownstreamTrack &t3 = dsTracks[vertices3[autopassVtx].GetTrackIndex(2)];

      SpectrometerTrackVertex &vtx = vertices3[k3piVertex];
      float vertexTime = vtx.GetTime();
      flat.fVertex.fX = vtx.GetPosition().X();
      flat.fVertex.fY = vtx.GetPosition().Y();
      flat.fVertex.fZ = vtx.GetPosition().Z();
      FillTrack(t1, flat.track1, vertexTime);
      FillTrack(t2, flat.track2, vertexTime);
      FillTrack(t3, flat.track3, vertexTime);
    }
    else{ // Necessarily autopassTrack is set
      DownstreamTrack &t1 = dsTracks[autopassTrack];
      float trackTime = t1.GetMostPreciseTime();

      flat.fVertex.fX = t1.GetBeamAxisVertex().X();
      flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
      flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
      FillTrack(t1, flat.track1, trackTime);
    }

    // Now check presence of additional clusters
    auto clusters = *GetOutput<std::vector<EnergyCluster>>("EnergyClusterBuilder.Output");
    std::vector<int> goodClusters = additionalClusters(clusters);
    if(goodClusters.size()>=1)
      FillCluster(clusters[goodClusters[0]], flat.clus1);
    if(goodClusters.size()>=2)
      FillCluster(clusters[goodClusters[1]], flat.clus2);

  }
  else if(k3pi_sel) {
    fEventType = 1;
    DownstreamTrack &t1 = dsTracks[vertices3[k3piVertex].GetTrackIndex(0)];
    DownstreamTrack &t2 = dsTracks[vertices3[k3piVertex].GetTrackIndex(1)];
    DownstreamTrack &t3 = dsTracks[vertices3[k3piVertex].GetTrackIndex(2)];

    SpectrometerTrackVertex &vtx = vertices3[k3piVertex];
    float vertexTime = vtx.GetTime();
    flat.fVertex.fX = vtx.GetPosition().X();
    flat.fVertex.fY = vtx.GetPosition().Y();
    flat.fVertex.fZ = vtx.GetPosition().Z();
    FillTrack(t1, flat.track1, vertexTime);
    FillTrack(t2, flat.track2, vertexTime);
    FillTrack(t3, flat.track3, vertexTime);
  }
  else if(ke3_sel) {
    fEventType = 2;
    DownstreamTrack &t1 = dsTracks[ke3Track];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = t1.GetBeamAxisVertex().X();
    flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
    flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
    FillTrack(t1, flat.track1, trackTime);

    auto clusters = *GetOutput<std::vector<EnergyCluster>>("EnergyClusterBuilder.Output");
    FillCluster(clusters[ke3Pi0.fClustersID.first], flat.clus1);
    FillCluster(clusters[ke3Pi0.fClustersID.second], flat.clus2);
  }
  else if(kmu2_sel) {
    fEventType = 3;
    DownstreamTrack &t1 = dsTracks[kmu2Track];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = t1.GetBeamAxisVertex().X();
    flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
    flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
    FillTrack(t1, flat.track1, trackTime);
  }
  else if(k2pi_sel) {
    fEventType = 4;
    DownstreamTrack &t1 = dsTracks[k2piTrack];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = k2piVtx.X();
    flat.fVertex.fY = k2piVtx.Y();
    flat.fVertex.fZ = k2piVtx.Z();
    FillTrack(t1, flat.track1, trackTime);
    FillCluster(k2piPi0.fEnergyClusters.first, flat.clus1);
    FillCluster(k2piPi0.fEnergyClusters.second, flat.clus2);
  }
  else if(kmu3_sel) {
    fEventType = 2;
    DownstreamTrack &t1 = dsTracks[kmu3Track];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = t1.GetBeamAxisVertex().X();
    flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
    flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
    FillTrack(t1, flat.track1, trackTime);

    auto clusters = *GetOutput<std::vector<EnergyCluster>>("EnergyClusterBuilder.Output");
    FillCluster(clusters[kmu3Pi0.fClustersID.first], flat.clus1);
    FillCluster(clusters[kmu3Pi0.fClustersID.second], flat.clus2);
  }
  myTree->Fill();
}

void export_flat::FillTrack(DownstreamTrack &t, TrackStruct &ts, float refTime){
  TVector3 mom = t.GetSpectrometerCandidate()->GetThreeMomentumBeforeMagnet();
  ts.fExists = true;
  ts.fDirectionX = mom.Unit().X();
  ts.fDirectionY = mom.Unit().Y();
  ts.fDirectionZ = mom.Unit().Z();
  ts.fMomentumMag = t.GetMomentum();
  ts.fTime = t.GetMostPreciseTime();
  ts.fLKrEnergy = t.GetLKrEnergy();
  ts.fRICHMLH = t.GetRICHMostLikelyHypothesis();
  ts.fRICHNHits = t.GetRICHNumberOfInTimeHits();
  ts.fRICHRadius = t.GetRICHRingRadius();
  ts.fCharge = t.GetCharge();
  TVector2 richCenter = t.GetRICHRingCentrePosition();
  ts.fRICHCenterX = richCenter.X();
  ts.fRICHCenterY = richCenter.Y();
  for(int i = 0; i < t.GetNMUV3AssociationRecords(); ++i){
    FillHisto("muv3tdiff", t.GetMUV3Time(i)-ts.fTime);
  }
  ts.fHasMUV3 = t.GetNMUV3InTimeAssociationRecords(refTime, 1.5)>0;
}

void export_flat::FillCluster(EnergyCluster &c, GammaStruct &ts){
  ts.fExists = true;
  ts.fLKrEnergy = c.GetEnergy();
  ts.fPositionX = c.GetX();
  ts.fPositionY = c.GetY();
  ts.fTime = c.GetTime();
}

Int_t export_flat::bestInTimeVertices(std::vector<SpectrometerTrackVertex> &vtc) {
  TRecoCedarEvent *cedar = GetEvent<TRecoCedarEvent>();
  Int_t iBest = -1;
  Double_t dtBest = 9999.;
  Double_t timeBest = -99.;
  for(Int_t iCand=0; iCand<cedar->GetNCandidates(); ++iCand) {
    Double_t dt = fabs(cedar->GetCandidate(iCand)->GetTime() - fReferenceTime);
    if (dt < dtBest && dt < 5){ // At most 5ns from the trigger
      dtBest = dt;
      iBest = iCand;
      timeBest = cedar->GetCandidate(iCand)->GetTime();
    }
  }

  std::vector<SpectrometerTrackVertex> ret;
  iBest = -1;
  dtBest = 9999.;
  Int_t iv = 0;
  for(auto vtx : vtc) {
    Double_t dt = fabs(vtx.GetTime() - timeBest);
    if(dt < dtBest && dt < 5){ // At most 5ns from the best KTAG
      dtBest = dt;
      iBest = iv;
    }
    ++iv;
  }

  return iBest;
}

std::vector<int> export_flat::goodDSTracks(std::vector<DownstreamTrack> &ds) {

  std::vector<int> goodTracks;
  if(ds.size() > 10)
    return goodTracks;

  Int_t trackID = 0;
  for(auto &track : ds) {
    if(!track.CHODTimeExists() && !track.NewCHODAssociationExists())
      continue;

    Double_t TrackTime =
      (track.CHODTimeExists()) ? track.GetCHODTime() : track.GetNewCHODTime();
    TrackTime -= fReferenceTime;  // with respect to the trigger time
    if(fabs(TrackTime) > 10.0)
      continue;  // out of time with the trigger

    if(track.GetNChambers() != 4)
      continue;
    if(track.GetChi2() > 20.0)
      continue;

    goodTracks.push_back(trackID++);
  }

  return goodTracks;
}

Bool_t export_flat::autopassSelection(std::vector<SpectrometerTrackVertex> &vertex3, std::vector<DownstreamTrack> &dsTracks, Int_t &goodvtx, Int_t &goodtrack) {
  if(++fDSCount<fDownscaling) return false;

  fDSCount = 0;

  goodvtx = bestInTimeVertices(vertex3);
  if(goodvtx!=-1) return true;

  std::vector<int> goodTracks = goodDSTracks(dsTracks);
  if(goodTracks.size()==1) {
    goodtrack = goodTracks[0];
    return true;
  }

  return false;
}

std::vector<int> export_flat::additionalClusters(std::vector<EnergyCluster> &clusters) {
  std::vector<int> goodClusters;
  Int_t iClus = 0;
  for(const auto &clus: clusters) {
    if((fabs(clus.GetTime() - fReferenceTime) > 20))
      continue;
    if((clus.GetDDeadCell() < 20.))
      continue;
    if((!clus.GetIsElectromagnetic()))
      continue;
    if((clus.GetEnergy() < 1000.))
      continue;
    if(clus.SpectrometerAssociationExists())
      continue;
    if(clus.GetHasCHODAssociation())
      continue;
    if(!clus.GetIsIsolated())
      continue;
    goodClusters.push_back(iClus++);
  }

  return goodClusters;
}

void export_flat::PostProcess() {
}

void export_flat::EndOfBurstUser() {
}

void export_flat::EndOfRunUser() {
}

void export_flat::EndOfJobUser() {
  myTree->Write();
  SaveAllPlots();
}

void export_flat::DrawPlot() {
}

export_flat::~export_flat() {
}
