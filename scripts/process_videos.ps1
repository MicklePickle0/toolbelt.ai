# Full pipeline over several videos, one after another (detection is the slow part).
# find_play first, so ball detection skips the time between rallies. Detection resumes where it stopped, so the run can
# be interrupted (Ctrl+C) and started again later. Log: data/outputs/logs/process_videos.log
#
# Usage (from the repo root, in PowerShell):
#   .\scripts\process_videos.ps1                       # the default list below
#   .\scripts\process_videos.ps1 leaside-panther_set1 niagara-rtw_set1

param([string[]]$Videos = @("leaside-panther_set2", "mac-iron_set2", "titan-rise_set2"))

$py = ".venv\Scripts\python.exe"
$log = "data\outputs\logs\process_videos.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
# find_rallies twice: check_blocks needs its point winners, and the second run adds the
# stuff blocks it found.
# track_camera needs the video calibrated first (web app, or scripts/calibrate.py).
$steps = "find_play", "detect_ball", "detect_people", "track_camera", "track_ball",
         "track_people", "pose_people", "find_rallies", "check_blocks", "find_rallies"

foreach ($v in $Videos) {
    foreach ($step in $steps) {
        "$(Get-Date -Format HH:mm) $v $step" | Tee-Object -FilePath $log -Append
        & $py "scripts\$step.py" --video $v *>> $log
        if ($LASTEXITCODE -ne 0) {
            "$(Get-Date -Format HH:mm) $v FAILED at $step (see log); next video" |
                Tee-Object -FilePath $log -Append
            break
        }
    }
}
"$(Get-Date -Format HH:mm) all done" | Tee-Object -FilePath $log -Append
