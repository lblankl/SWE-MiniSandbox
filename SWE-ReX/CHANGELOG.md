# Changelog

## 1.2.1 (03/04/25)

### Fixed

* `TemporaryDirectory` cleanup issues on Windows in https://github.com/SWE-agent/SWE-ReX/pull/181

### Enhanced

* Message about building docker images in https://github.com/SWE-agent/SWE-ReX/pull/178

**Full Changelog**: https://github.com/SWE-agent/SWE-ReX/compare/v1.2.0...v1.2.1

## 1.2.0 (02/25/25)

### Added

* Add platform flag to Deployment config for docker builds on different platforms by @carlosejimenez in https://github.com/SWE-agent/SWE-ReX/pull/165

### Fixed

* Fix: Avoid exceptions from Deployment.__del__ by @klieret in https://github.com/SWE-agent/SWE-ReX/pull/170
* Enh: Validate image ID by @klieret in https://github.com/SWE-agent/SWE-ReX/pull/173

**Full Changelog**: https://github.com/SWE-agent/SWE-ReX/compare/v1.1.1...v1.2.0

## 1.1.1 (02/15/25)

### Added

* Added `swerex.utils.aws_teardown` utility to clean up AWS resources created by the Fargate deployment

## 1.1.0 (01/25/25)

### Added

* Added `encoding` and `error` arguments to `read_file`

## 1.0.5

### Fixes

* Catching more errors from bashlex failures

## 1.0.4

### Fixes

* Fixed bugs around unicode decode errors
