; Inno Setup script of the Windows installer (built by scripts/build_installer.py).
; Installs the PyInstaller folder into Program Files (or the user's folder without admin rights),
; puts dlnet.exe on PATH and adds Start-menu entries for the MCP server.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\dlnet"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist\installers"
#endif

[Setup]
AppId={{9C705120-8530-4CF0-BFEE-A6978C89D552}
AppName=DLNet
AppVersion={#AppVersion}
AppPublisher=DLNet
DefaultDirName={autopf}\DLNet
DefaultGroupName=DLNet
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename=DLNet-{#AppVersion}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes
UninstallDisplayName=DLNet {#AppVersion}
WizardStyle=modern

[Tasks]
Name: "addtopath"; Description: "Add dlnet to PATH"; Flags: checkedonce

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DLNet MCP server"; Filename: "{app}\dlnet.exe"; Parameters: "serve --workspace ""{userdocs}\dlnet-workspace"""; WorkingDir: "{userdocs}"
Name: "{group}\DLNet command prompt"; Filename: "{cmd}"; Parameters: "/K ""set PATH={app};%PATH% && dlnet --help"""; WorkingDir: "{userdocs}"
Name: "{group}\Uninstall DLNet"; Filename: "{uninstallexe}"

[Registry]
Root: HKA; Subkey: "{code:EnvKey}"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function EnvKey(Param: String): String;
begin
  if IsAdminInstallMode then
    Result := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
  else
    Result := 'Environment';
end;

function NeedsAddPath(Dir: String): Boolean;
var
  Paths: String;
  Root: Integer;
begin
  if IsAdminInstallMode then Root := HKEY_LOCAL_MACHINE else Root := HKEY_CURRENT_USER;
  if not RegQueryStringValue(Root, EnvKey(''), 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Paths, Dir: String;
  Root, P: Integer;
begin
  if CurUninstallStep <> usPostUninstall then exit;
  if IsAdminInstallMode then Root := HKEY_LOCAL_MACHINE else Root := HKEY_CURRENT_USER;
  if not RegQueryStringValue(Root, EnvKey(''), 'Path', Paths) then exit;
  Dir := ';' + ExpandConstant('{app}');
  P := Pos(Uppercase(Dir), Uppercase(Paths));
  if P > 0 then
  begin
    Delete(Paths, P, Length(Dir));
    RegWriteExpandStringValue(Root, EnvKey(''), 'Path', Paths);
  end;
end;
