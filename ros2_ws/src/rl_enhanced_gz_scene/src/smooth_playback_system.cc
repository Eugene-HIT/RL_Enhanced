#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Types.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <sdf/Element.hh>

namespace rl_enhanced_gz_scene
{
struct TrajSample
{
  double t{0.0};
  gz::math::Vector3d p{0.0, 0.0, 0.0};
  double theta{0.0};
  bool hasDronePositions{false};
  gz::math::Vector3d droneFront{0.0, 0.0, 0.0};
  gz::math::Vector3d droneRearLeft{0.0, 0.0, 0.0};
  gz::math::Vector3d droneRearRight{0.0, 0.0, 0.0};
};

class SmoothPlaybackSystem final : public gz::sim::System,
                                   public gz::sim::ISystemConfigure,
                                   public gz::sim::ISystemPreUpdate
{
  public: void Configure(
      const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &,
      gz::sim::EventManager &) override
  {
    this->enabled = false;
    if (_sdf && _sdf->HasElement("enabled"))
      this->enabled = _sdf->Get<bool>("enabled");

    if (const char *envEnable = std::getenv("RL_ENHANCED_SMOOTH_ENABLE"))
      this->enabled = std::string(envEnable) != "0";

    this->csvPath = "/tmp/rl_enhanced_playback_samples.csv";
    if (_sdf && _sdf->HasElement("csv_path"))
      this->csvPath = _sdf->Get<std::string>("csv_path");
    if (const char *envCsv = std::getenv("RL_ENHANCED_PLAYBACK_CSV"))
      this->csvPath = std::string(envCsv);

    if (_sdf && _sdf->HasElement("payload_name"))
      this->payloadName = _sdf->Get<std::string>("payload_name");
    if (_sdf && _sdf->HasElement("goal_marker_name"))
      this->goalMarkerName = _sdf->Get<std::string>("goal_marker_name");
    if (_sdf && _sdf->HasElement("drone_front_name"))
      this->droneFrontName = _sdf->Get<std::string>("drone_front_name");
    if (_sdf && _sdf->HasElement("drone_rear_left_name"))
      this->droneRearLeftName = _sdf->Get<std::string>("drone_rear_left_name");
    if (_sdf && _sdf->HasElement("drone_rear_right_name"))
      this->droneRearRightName = _sdf->Get<std::string>("drone_rear_right_name");
    if (_sdf && _sdf->HasElement("arrow_front_head_name"))
      this->arrowFrontHeadName = _sdf->Get<std::string>("arrow_front_head_name");
    if (_sdf && _sdf->HasElement("arrow_rear_left_head_name"))
      this->arrowRearLeftHeadName = _sdf->Get<std::string>("arrow_rear_left_head_name");
    if (_sdf && _sdf->HasElement("arrow_rear_right_head_name"))
      this->arrowRearRightHeadName = _sdf->Get<std::string>("arrow_rear_right_head_name");

    if (_sdf && _sdf->HasElement("speed"))
      this->speed = _sdf->Get<double>("speed");
    if (_sdf && _sdf->HasElement("z_offset"))
      this->zOffset = _sdf->Get<double>("z_offset");
    if (_sdf && _sdf->HasElement("drone_z_rel"))
      this->droneZRel = _sdf->Get<double>("drone_z_rel");
    if (_sdf && _sdf->HasElement("goal_marker_yaw_rate"))
      this->goalMarkerYawRate = _sdf->Get<double>("goal_marker_yaw_rate");

    if (const char *envSpeed = std::getenv("RL_ENHANCED_SMOOTH_SPEED"))
    {
      try
      {
        this->speed = std::stod(envSpeed);
      }
      catch (const std::exception &)
      {
      }
    }
    if (const char *envZOffset = std::getenv("RL_ENHANCED_SMOOTH_Z_OFFSET"))
    {
      try
      {
        this->zOffset = std::stod(envZOffset);
      }
      catch (const std::exception &)
      {
      }
    }

    if (_sdf && _sdf->HasElement("goal_marker_x"))
      this->goalMarkerPos.X(_sdf->Get<double>("goal_marker_x"));
    if (_sdf && _sdf->HasElement("goal_marker_y"))
      this->goalMarkerPos.Y(_sdf->Get<double>("goal_marker_y"));
    if (_sdf && _sdf->HasElement("goal_marker_z"))
      this->goalMarkerPos.Z(_sdf->Get<double>("goal_marker_z"));

    if (_sdf && _sdf->HasElement("drone_front_x"))
      this->droneFrontOffset.X(_sdf->Get<double>("drone_front_x"));
    if (_sdf && _sdf->HasElement("drone_front_y"))
      this->droneFrontOffset.Y(_sdf->Get<double>("drone_front_y"));

    if (_sdf && _sdf->HasElement("drone_rear_left_x"))
      this->droneRearLeftOffset.X(_sdf->Get<double>("drone_rear_left_x"));
    if (_sdf && _sdf->HasElement("drone_rear_left_y"))
      this->droneRearLeftOffset.Y(_sdf->Get<double>("drone_rear_left_y"));

    if (_sdf && _sdf->HasElement("drone_rear_right_x"))
      this->droneRearRightOffset.X(_sdf->Get<double>("drone_rear_right_x"));
    if (_sdf && _sdf->HasElement("drone_rear_right_y"))
      this->droneRearRightOffset.Y(_sdf->Get<double>("drone_rear_right_y"));

    if (_sdf && _sdf->HasElement("attach_front_x"))
      this->attachFrontOffset.X(_sdf->Get<double>("attach_front_x"));
    if (_sdf && _sdf->HasElement("attach_front_y"))
      this->attachFrontOffset.Y(_sdf->Get<double>("attach_front_y"));
    if (_sdf && _sdf->HasElement("attach_front_z"))
      this->attachFrontOffset.Z(_sdf->Get<double>("attach_front_z"));
    if (_sdf && _sdf->HasElement("attach_rear_left_x"))
      this->attachRearLeftOffset.X(_sdf->Get<double>("attach_rear_left_x"));
    if (_sdf && _sdf->HasElement("attach_rear_left_y"))
      this->attachRearLeftOffset.Y(_sdf->Get<double>("attach_rear_left_y"));
    if (_sdf && _sdf->HasElement("attach_rear_left_z"))
      this->attachRearLeftOffset.Z(_sdf->Get<double>("attach_rear_left_z"));
    if (_sdf && _sdf->HasElement("attach_rear_right_x"))
      this->attachRearRightOffset.X(_sdf->Get<double>("attach_rear_right_x"));
    if (_sdf && _sdf->HasElement("attach_rear_right_y"))
      this->attachRearRightOffset.Y(_sdf->Get<double>("attach_rear_right_y"));
    if (_sdf && _sdf->HasElement("attach_rear_right_z"))
      this->attachRearRightOffset.Z(_sdf->Get<double>("attach_rear_right_z"));

    this->loaded = this->LoadCsv(this->csvPath);
    if (!this->enabled)
    {
      std::cout << "[smooth_playback] disabled" << std::endl;
      return;
    }

    if (!this->loaded)
      std::cerr << "[smooth_playback] failed to load csv: " << this->csvPath << std::endl;
    else
      std::cout << "[smooth_playback] enabled, samples=" << this->samples.size() << std::endl;
  }

  public: void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (!this->enabled || !this->loaded)
      return;

    if (!this->started && !_info.paused)
    {
      this->simStartTime = _info.simTime;
      this->started = true;
    }

    this->ResolveEntities(_ecm);

    double trajTime = 0.0;
    if (this->started)
    {
      const double simSec =
        std::chrono::duration<double>(_info.simTime - this->simStartTime).count() *
        this->speed;
      trajTime = std::clamp(simSec, 0.0, this->samples.back().t);
    }

    gz::math::Vector3d pos;
    double theta = 0.0;
    bool hasDronePositions = false;
    gz::math::Vector3d droneFrontPos;
    gz::math::Vector3d droneRearLeftPos;
    gz::math::Vector3d droneRearRightPos;
    this->Interpolate(
      trajTime,
      pos,
      theta,
      hasDronePositions,
      droneFrontPos,
      droneRearLeftPos,
      droneRearRightPos);
    pos.Z(pos.Z() + this->zOffset);
    if (hasDronePositions)
    {
      droneFrontPos.Z(droneFrontPos.Z() + this->zOffset);
      droneRearLeftPos.Z(droneRearLeftPos.Z() + this->zOffset);
      droneRearRightPos.Z(droneRearRightPos.Z() + this->zOffset);
    }

    const gz::math::Pose3d payloadPose(pos.X(), pos.Y(), pos.Z(), 0.0, theta, 0.0);
    this->SetPoseCmd(this->payloadEntity, payloadPose, _ecm);

    const gz::math::Quaterniond payloadRotation(0.0, theta, 0.0);

    if (hasDronePositions)
    {
      this->SetPoseCmd(
        this->droneFrontEntity,
        gz::math::Pose3d(
          droneFrontPos.X(),
          droneFrontPos.Y(),
          droneFrontPos.Z(),
          0.0,
          0.0,
          0.0),
        _ecm);

      this->SetPoseCmd(
        this->droneRearLeftEntity,
        gz::math::Pose3d(
          droneRearLeftPos.X(),
          droneRearLeftPos.Y(),
          droneRearLeftPos.Z(),
          0.0,
          0.0,
          0.60),
        _ecm);

      this->SetPoseCmd(
        this->droneRearRightEntity,
        gz::math::Pose3d(
          droneRearRightPos.X(),
          droneRearRightPos.Y(),
          droneRearRightPos.Z(),
          0.0,
          0.0,
          -0.60),
        _ecm);
    }
    else
    {
      this->SetPoseCmd(
          this->droneFrontEntity,
          gz::math::Pose3d(
              pos.X() + this->droneFrontOffset.X(),
              pos.Y() + this->droneFrontOffset.Y(),
              pos.Z() + this->droneZRel,
              0.0,
              0.0,
              0.0),
          _ecm);

      this->SetPoseCmd(
          this->droneRearLeftEntity,
          gz::math::Pose3d(
              pos.X() + this->droneRearLeftOffset.X(),
              pos.Y() + this->droneRearLeftOffset.Y(),
              pos.Z() + this->droneZRel,
              0.0,
              0.0,
              0.60),
          _ecm);

      this->SetPoseCmd(
          this->droneRearRightEntity,
          gz::math::Pose3d(
              pos.X() + this->droneRearRightOffset.X(),
              pos.Y() + this->droneRearRightOffset.Y(),
              pos.Z() + this->droneZRel,
              0.0,
              0.0,
              -0.60),
          _ecm);

      droneFrontPos = gz::math::Vector3d(
        pos.X() + this->droneFrontOffset.X(),
        pos.Y() + this->droneFrontOffset.Y(),
        pos.Z() + this->droneZRel);
      droneRearLeftPos = gz::math::Vector3d(
        pos.X() + this->droneRearLeftOffset.X(),
        pos.Y() + this->droneRearLeftOffset.Y(),
        pos.Z() + this->droneZRel);
      droneRearRightPos = gz::math::Vector3d(
        pos.X() + this->droneRearRightOffset.X(),
        pos.Y() + this->droneRearRightOffset.Y(),
        pos.Z() + this->droneZRel);
    }

    const gz::math::Vector3d attachFrontWorld = pos + payloadRotation.RotateVector(this->attachFrontOffset);
    const gz::math::Vector3d attachRearLeftWorld = pos + payloadRotation.RotateVector(this->attachRearLeftOffset);
    const gz::math::Vector3d attachRearRightWorld = pos + payloadRotation.RotateVector(this->attachRearRightOffset);
    this->UpdateArrow(
      this->arrowFrontSegmentEntities,
      this->arrowFrontHeadEntity,
      attachFrontWorld,
      droneFrontPos,
      _ecm);
    this->UpdateArrow(
      this->arrowRearLeftSegmentEntities,
      this->arrowRearLeftHeadEntity,
      attachRearLeftWorld,
      droneRearLeftPos,
      _ecm);
    this->UpdateArrow(
      this->arrowRearRightSegmentEntities,
      this->arrowRearRightHeadEntity,
      attachRearRightWorld,
      droneRearRightPos,
      _ecm);
    this->SetPoseCmd(this->attachFrontMarkerEntity, gz::math::Pose3d(attachFrontWorld, gz::math::Quaterniond::Identity), _ecm);
    this->SetPoseCmd(this->attachRearLeftMarkerEntity, gz::math::Pose3d(attachRearLeftWorld, gz::math::Quaterniond::Identity), _ecm);
    this->SetPoseCmd(this->attachRearRightMarkerEntity, gz::math::Pose3d(attachRearRightWorld, gz::math::Quaterniond::Identity), _ecm);

    if (this->goalMarkerEntity != gz::sim::kNullEntity)
    {
      const double markerYaw = this->goalMarkerYawRate * trajTime;
      const gz::math::Pose3d markerPose(
          this->goalMarkerPos.X(),
          this->goalMarkerPos.Y(),
          this->goalMarkerPos.Z(),
          0.0,
          0.0,
          markerYaw);
      this->SetPoseCmd(this->goalMarkerEntity, markerPose, _ecm);
    }
  }

  private: bool LoadCsv(const std::string &_path)
  {
    std::ifstream ifs(_path);
    if (!ifs.is_open())
      return false;

    this->samples.clear();
    std::string line;
    while (std::getline(ifs, line))
    {
      if (line.empty() || line[0] == '#')
        continue;
      std::replace(line.begin(), line.end(), ';', ',');
      std::stringstream ss(line);
      std::string tok;
      std::vector<double> vals;
      while (std::getline(ss, tok, ','))
      {
        try
        {
          vals.push_back(std::stod(tok));
        }
        catch (const std::exception &)
        {
          vals.clear();
          break;
        }
      }
      if (vals.size() < 5)
        continue;
      TrajSample s;
      s.t = vals[0];
      s.p.Set(vals[1], vals[2], vals[3]);
      s.theta = vals[4];
      if (vals.size() >= 14)
      {
        s.hasDronePositions = true;
        s.droneFront.Set(vals[5], vals[6], vals[7]);
        s.droneRearLeft.Set(vals[8], vals[9], vals[10]);
        s.droneRearRight.Set(vals[11], vals[12], vals[13]);
      }
      this->samples.push_back(s);
    }

    return this->samples.size() >= 2;
  }

  private: void ResolveEntities(gz::sim::EntityComponentManager &_ecm)
  {
    if (this->payloadEntity == gz::sim::kNullEntity)
      this->payloadEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->payloadName),
              gz::sim::components::Model());
    if (this->droneFrontEntity == gz::sim::kNullEntity)
      this->droneFrontEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->droneFrontName),
              gz::sim::components::Model());
    if (this->droneRearLeftEntity == gz::sim::kNullEntity)
      this->droneRearLeftEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->droneRearLeftName),
              gz::sim::components::Model());
    if (this->droneRearRightEntity == gz::sim::kNullEntity)
      this->droneRearRightEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->droneRearRightName),
              gz::sim::components::Model());
    if (this->goalMarkerEntity == gz::sim::kNullEntity)
      this->goalMarkerEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->goalMarkerName),
              gz::sim::components::Model());
    for (size_t idx = 0; idx < this->arrowFrontSegmentNames.size(); ++idx)
    {
      if (this->arrowFrontSegmentEntities[idx] == gz::sim::kNullEntity)
        this->arrowFrontSegmentEntities[idx] =
            _ecm.EntityByComponents(
                gz::sim::components::Name(this->arrowFrontSegmentNames[idx]),
                gz::sim::components::Model());
      if (this->arrowRearLeftSegmentEntities[idx] == gz::sim::kNullEntity)
        this->arrowRearLeftSegmentEntities[idx] =
            _ecm.EntityByComponents(
                gz::sim::components::Name(this->arrowRearLeftSegmentNames[idx]),
                gz::sim::components::Model());
      if (this->arrowRearRightSegmentEntities[idx] == gz::sim::kNullEntity)
        this->arrowRearRightSegmentEntities[idx] =
            _ecm.EntityByComponents(
                gz::sim::components::Name(this->arrowRearRightSegmentNames[idx]),
                gz::sim::components::Model());
    }
    if (this->arrowFrontHeadEntity == gz::sim::kNullEntity)
      this->arrowFrontHeadEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->arrowFrontHeadName),
              gz::sim::components::Model());
    if (this->arrowRearLeftHeadEntity == gz::sim::kNullEntity)
      this->arrowRearLeftHeadEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->arrowRearLeftHeadName),
              gz::sim::components::Model());
    if (this->arrowRearRightHeadEntity == gz::sim::kNullEntity)
      this->arrowRearRightHeadEntity =
          _ecm.EntityByComponents(
              gz::sim::components::Name(this->arrowRearRightHeadName),
              gz::sim::components::Model());
    if (this->attachFrontMarkerEntity == gz::sim::kNullEntity)
      this->attachFrontMarkerEntity =
        _ecm.EntityByComponents(
          gz::sim::components::Name(this->attachFrontMarkerName),
          gz::sim::components::Model());
    if (this->attachRearLeftMarkerEntity == gz::sim::kNullEntity)
      this->attachRearLeftMarkerEntity =
        _ecm.EntityByComponents(
          gz::sim::components::Name(this->attachRearLeftMarkerName),
          gz::sim::components::Model());
    if (this->attachRearRightMarkerEntity == gz::sim::kNullEntity)
      this->attachRearRightMarkerEntity =
        _ecm.EntityByComponents(
          gz::sim::components::Name(this->attachRearRightMarkerName),
          gz::sim::components::Model());
  }

  private: gz::math::Quaterniond RotationFromZAxis(const gz::math::Vector3d &_direction) const
  {
    const gz::math::Vector3d zAxis(0.0, 0.0, 1.0);
    const double dot = std::clamp(zAxis.Dot(_direction), -1.0, 1.0);
    if (dot > 1.0 - 1e-9)
      return gz::math::Quaterniond::Identity;
    if (dot < -1.0 + 1e-9)
      return gz::math::Quaterniond(0.0, GZ_PI, 0.0);

    gz::math::Vector3d axis = zAxis.Cross(_direction);
    axis.Normalize();
    return gz::math::Quaterniond(axis, std::acos(dot));
  }

  private: void UpdateArrow(
      const std::array<gz::sim::Entity, 3> &_segmentEntities,
      const gz::sim::Entity _headEntity,
      const gz::math::Vector3d &_start,
      const gz::math::Vector3d &_end,
      gz::sim::EntityComponentManager &_ecm)
  {
    const gz::math::Vector3d delta = _end - _start;
    const double length = delta.Length();
    if (length < 1e-6)
      return;

    gz::math::Vector3d direction = delta;
    direction.Normalize();
    const gz::math::Quaterniond rotation = this->RotationFromZAxis(direction);
    const std::array<double, 3> fractions{0.22, 0.48, 0.72};
    for (size_t idx = 0; idx < _segmentEntities.size(); ++idx)
    {
      const gz::math::Vector3d pos = _start + fractions[idx] * delta;
      this->SetPoseCmd(_segmentEntities[idx], gz::math::Pose3d(pos, rotation), _ecm);
    }
    const gz::math::Vector3d headPos = _start + 0.90 * delta;
    this->SetPoseCmd(_headEntity, gz::math::Pose3d(headPos, rotation), _ecm);
  }

  private: void SetPoseCmd(
      const gz::sim::Entity _entity,
      const gz::math::Pose3d &_pose,
      gz::sim::EntityComponentManager &_ecm)
  {
    if (_entity == gz::sim::kNullEntity)
      return;

    auto *poseCmd = _ecm.Component<gz::sim::components::WorldPoseCmd>(_entity);
    if (!poseCmd)
    {
      _ecm.CreateComponent(_entity, gz::sim::components::WorldPoseCmd(_pose));
      return;
    }
    _ecm.SetComponentData<gz::sim::components::WorldPoseCmd>(_entity, _pose);
  }

  private: void Interpolate(
      const double _t,
      gz::math::Vector3d &_pos,
      double &_theta,
      bool &_hasDronePositions,
      gz::math::Vector3d &_droneFront,
      gz::math::Vector3d &_droneRearLeft,
      gz::math::Vector3d &_droneRearRight) const
  {
    if (_t <= this->samples.front().t)
    {
      _pos = this->samples.front().p;
      _theta = this->samples.front().theta;
      _hasDronePositions = this->samples.front().hasDronePositions;
      _droneFront = this->samples.front().droneFront;
      _droneRearLeft = this->samples.front().droneRearLeft;
      _droneRearRight = this->samples.front().droneRearRight;
      return;
    }
    if (_t >= this->samples.back().t)
    {
      _pos = this->samples.back().p;
      _theta = this->samples.back().theta;
      _hasDronePositions = this->samples.back().hasDronePositions;
      _droneFront = this->samples.back().droneFront;
      _droneRearLeft = this->samples.back().droneRearLeft;
      _droneRearRight = this->samples.back().droneRearRight;
      return;
    }

    size_t hi = 1;
    while (hi < this->samples.size() && this->samples[hi].t < _t)
      ++hi;
    const size_t lo = hi - 1;
    const auto &a = this->samples[lo];
    const auto &b = this->samples[hi];

    const double dt = std::max(1e-9, b.t - a.t);
    const double u = std::clamp((_t - a.t) / dt, 0.0, 1.0);
    _pos = (1.0 - u) * a.p + u * b.p;
    _hasDronePositions = a.hasDronePositions && b.hasDronePositions;
    if (_hasDronePositions)
    {
      _droneFront = (1.0 - u) * a.droneFront + u * b.droneFront;
      _droneRearLeft = (1.0 - u) * a.droneRearLeft + u * b.droneRearLeft;
      _droneRearRight = (1.0 - u) * a.droneRearRight + u * b.droneRearRight;
    }

    const double d = std::atan2(std::sin(b.theta - a.theta), std::cos(b.theta - a.theta));
    _theta = a.theta + u * d;
  }

  private: bool enabled{false};
  private: bool loaded{false};
  private: bool started{false};
  private: std::chrono::steady_clock::duration simStartTime{0};
  private: std::string csvPath{"/tmp/rl_enhanced_playback_samples.csv"};

  private: std::string payloadName{"trajectory_payload"};
  private: std::string droneFrontName{"world_drone_front"};
  private: std::string droneRearLeftName{"world_drone_rear_left"};
  private: std::string droneRearRightName{"world_drone_rear_right"};
  private: std::string arrowFrontHeadName{"arrow_front_head"};
  private: std::string arrowRearLeftHeadName{"arrow_rear_left_head"};
  private: std::string arrowRearRightHeadName{"arrow_rear_right_head"};
  private: std::string attachFrontMarkerName{"attach_front_marker"};
  private: std::string attachRearLeftMarkerName{"attach_rear_left_marker"};
  private: std::string attachRearRightMarkerName{"attach_rear_right_marker"};
  private: std::string goalMarkerName{"goal_cross_marker"};

  private: double speed{1.0};
  private: double zOffset{1.10};
  private: double droneZRel{0.40};
  private: double goalMarkerYawRate{0.30};
  private: gz::math::Vector3d goalMarkerPos{0.0, 19.0, 2.05};
  private: gz::math::Vector3d droneFrontOffset{0.20, 0.0, 0.0};
  private: gz::math::Vector3d droneRearLeftOffset{-0.17, 0.13, 0.0};
  private: gz::math::Vector3d droneRearRightOffset{-0.17, -0.13, 0.0};
  private: gz::math::Vector3d attachFrontOffset{-0.40, 0.0, -0.10};
  private: gz::math::Vector3d attachRearLeftOffset{0.40, 0.0, -0.10};
  private: gz::math::Vector3d attachRearRightOffset{0.0, 0.0, 0.10};
  private: std::array<std::string, 3> arrowFrontSegmentNames{"arrow_front_seg_0", "arrow_front_seg_1", "arrow_front_seg_2"};
  private: std::array<std::string, 3> arrowRearLeftSegmentNames{"arrow_rear_left_seg_0", "arrow_rear_left_seg_1", "arrow_rear_left_seg_2"};
  private: std::array<std::string, 3> arrowRearRightSegmentNames{"arrow_rear_right_seg_0", "arrow_rear_right_seg_1", "arrow_rear_right_seg_2"};

  private: std::vector<TrajSample> samples;
  private: gz::sim::Entity payloadEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity droneFrontEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity droneRearLeftEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity droneRearRightEntity{gz::sim::kNullEntity};
  private: std::array<gz::sim::Entity, 3> arrowFrontSegmentEntities{gz::sim::kNullEntity, gz::sim::kNullEntity, gz::sim::kNullEntity};
  private: std::array<gz::sim::Entity, 3> arrowRearLeftSegmentEntities{gz::sim::kNullEntity, gz::sim::kNullEntity, gz::sim::kNullEntity};
  private: std::array<gz::sim::Entity, 3> arrowRearRightSegmentEntities{gz::sim::kNullEntity, gz::sim::kNullEntity, gz::sim::kNullEntity};
  private: gz::sim::Entity arrowFrontHeadEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity arrowRearLeftHeadEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity arrowRearRightHeadEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity attachFrontMarkerEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity attachRearLeftMarkerEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity attachRearRightMarkerEntity{gz::sim::kNullEntity};
  private: gz::sim::Entity goalMarkerEntity{gz::sim::kNullEntity};
};
}  // namespace rl_enhanced_gz_scene

GZ_ADD_PLUGIN(
    rl_enhanced_gz_scene::SmoothPlaybackSystem,
    gz::sim::System,
    gz::sim::ISystemConfigure,
    gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(rl_enhanced_gz_scene::SmoothPlaybackSystem, "rl_enhanced_gz_scene::SmoothPlaybackSystem")
