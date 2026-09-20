// SPDX-License-Identifier: Apache-2.0
#include "file_dialog.h"

#include <stdexcept>
#include <string>

#include "paths.h"
#include <SDL3/SDL.h>
#ifdef _WIN32
#include <shobjidl.h>
#include <windows.h>
#include <wrl/client.h>
#endif

namespace din::studio
{
namespace
{
void Deliver(int action, const std::string& path)
{
    SDL_Event event{};
    event.type = SDL_EVENT_USER;
    event.user.code = action;
    event.user.data1 = SDL_strdup(path.c_str());
    if (!SDL_PushEvent(&event))
        SDL_free(event.user.data1);
}
#ifdef _WIN32
using Microsoft::WRL::ComPtr;
void Check(HRESULT hr)
{
    if (FAILED(hr))
        throw std::runtime_error("Windows file dialog failed (HRESULT " +
                                 std::to_string(static_cast<unsigned long>(hr)) + ")");
}
void DeliverItem(int action, IShellItem* item)
{
    PWSTR path = nullptr;
    Check(item->GetDisplayName(SIGDN_FILESYSPATH, &path));
    const std::filesystem::path result(path);
    CoTaskMemFree(path);
    Deliver(action, Utf8(result));
}
void WindowsDialog(HWND owner, int action, const std::filesystem::path& location, bool verify_only = false)
{
    Check(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE));
    struct Apartment
    {
        ~Apartment()
        {
            CoUninitialize();
        }
    } apartment;
    const int kind = action % 8;
    const bool save = kind == 3 || kind == 4;
    const bool folder = kind == 2 || kind == 6;
    ComPtr<IFileDialog> dialog;
    Check(CoCreateInstance(save ? CLSID_FileSaveDialog : CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                           IID_PPV_ARGS(dialog.GetAddressOf())));
    FILEOPENDIALOGOPTIONS options;
    Check(dialog->GetOptions(&options));
    options |= FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR | FOS_PATHMUSTEXIST;
    if (folder)
        options |= FOS_PICKFOLDERS;
    if (kind == 1 || kind == 5)
        options |= FOS_ALLOWMULTISELECT | FOS_FILEMUSTEXIST;
    if (save)
        options |= FOS_OVERWRITEPROMPT;
    Check(dialog->SetOptions(options));
    Check(dialog->SetTitle(kind == 5   ? L"Import videos"
                           : kind == 1 ? L"Import audio"
                           : kind == 2 ? L"Select model folder"
                           : kind == 6 ? L"Select forced aligner folder"
                                       : L"Export transcript"));
    if (!folder)
    {
        const COMDLG_FILTERSPEC filter =
            kind == 5   ? COMDLG_FILTERSPEC{L"Video files", L"*.mp4;*.mov;*.mkv;*.webm;*.avi;*.m4v"}
            : kind == 1 ? COMDLG_FILTERSPEC{L"Audio files", L"*.wav;*.mp3;*.flac"}
            : kind == 3 ? COMDLG_FILTERSPEC{L"Text", L"*.txt"}
                        : COMDLG_FILTERSPEC{L"JSON", L"*.json"};
        Check(dialog->SetFileTypes(1, &filter));
    }
    const auto directory = DialogFolder(save ? location.parent_path() : location);
    ComPtr<IShellItem> initial;
    Check(SHCreateItemFromParsingName(directory.c_str(), nullptr, IID_PPV_ARGS(initial.GetAddressOf())));
    Check(dialog->SetFolder(initial.Get()));
    if (save)
    {
        Check(dialog->SetFileName(location.filename().c_str()));
        Check(dialog->SetDefaultExtension(kind == 3 ? L"txt" : L"json"));
    }
    if (verify_only)
    {
        ComPtr<IShellItem> selected;
        Check(dialog->GetFolder(selected.GetAddressOf()));
        int comparison = 1;
        Check(initial->Compare(selected.Get(), SICHINT_CANONICAL, &comparison));
        if (comparison != 0)
            throw std::runtime_error("File dialog ignored its starting folder");
        return;
    }
    const auto result = dialog->Show(owner);
    if (result == HRESULT_FROM_WIN32(ERROR_CANCELLED))
        return;
    Check(result);
    if (kind == 1 || kind == 5)
    {
        ComPtr<IFileOpenDialog> open;
        Check(dialog->QueryInterface(IID_PPV_ARGS(open.GetAddressOf())));
        ComPtr<IShellItemArray> items;
        Check(open->GetResults(items.GetAddressOf()));
        DWORD count = 0;
        Check(items->GetCount(&count));
        for (DWORD i = 0; i < count; ++i)
        {
            ComPtr<IShellItem> item;
            Check(items->GetItemAt(i, item.GetAddressOf()));
            DeliverItem(action, item.Get());
        }
    }
    else
    {
        ComPtr<IShellItem> item;
        Check(dialog->GetResult(item.GetAddressOf()));
        DeliverItem(action, item.Get());
    }
}
#else
void SDLCALL FileChosen(void* data, const char* const* files, int)
{
    if (!files)
    {
        Deliver(-1, SDL_GetError());
        return;
    }
    const int action = static_cast<int>(reinterpret_cast<intptr_t>(data));
    for (size_t i = 0; files[i]; ++i)
        Deliver(action, files[i]);
}
#endif
}  // namespace
std::filesystem::path DialogFolder(std::filesystem::path path)
{
    std::error_code ec;
    if (path.empty())
        path = std::filesystem::current_path();
    path = std::filesystem::absolute(path).lexically_normal();
    while (!std::filesystem::is_directory(path, ec))
    {
        const auto parent = path.parent_path();
        if (parent == path || parent.empty())
            return std::filesystem::current_path();
        path = parent;
    }
    return path;
}
FileDialogs::~FileDialogs()
{
    if (thread_.joinable())
        thread_.join();
}
void VerifyFileDialogs(const std::filesystem::path& existing_file)
{
    const auto directory = std::filesystem::absolute(existing_file).parent_path();
    if (DialogFolder(existing_file) != directory || DialogFolder(existing_file / "missing" / "nested") != directory)
        throw std::runtime_error("Dialog folder fallback failed");
#ifdef _WIN32
    for (int action : {1, 2, 6, 3, 4})
        WindowsDialog(nullptr, action, existing_file, true);
#endif
}
void FileDialogs::Open(void* window, int action, const std::filesystem::path& location)
{
#ifdef _WIN32
    if (busy_.load())
        return;
    if (thread_.joinable())
        thread_.join();
    auto owner = static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(static_cast<SDL_Window*>(window)),
                                                          SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    busy_.store(true);
    try
    {
        thread_ = std::thread(
            [this, owner, action, location]
            {
                try
                {
                    WindowsDialog(owner, action, location);
                }
                catch (const std::exception& e)
                {
                    Deliver(-1, e.what());
                }
                busy_.store(false);
            });
    }
    catch (...)
    {
        busy_.store(false);
        throw;
    }
#else
    auto* w = static_cast<SDL_Window*>(window);
    auto* data = reinterpret_cast<void*>(static_cast<intptr_t>(action));
    const int kind = action % 8;
    const bool save = kind == 3 || kind == 4;
    const auto initial =
        Utf8(save ? DialogFolder(location.parent_path()) / location.filename() : DialogFolder(location));
    static const SDL_DialogFileFilter audio[] = {{"Audio", "wav;mp3;flac"}}, text[] = {{"Text", "txt"}},
                                      json[] = {{"JSON", "json"}};
    static const SDL_DialogFileFilter video[] = {{"Video", "mp4;mov;mkv;webm;avi;m4v"}};
    if (kind == 5)
        SDL_ShowOpenFileDialog(FileChosen, data, w, video, 1, initial.c_str(), true);
    else if (kind == 1)
        SDL_ShowOpenFileDialog(FileChosen, data, w, audio, 1, initial.c_str(), true);
    else if (kind == 2 || kind == 6)
        SDL_ShowOpenFolderDialog(FileChosen, data, w, initial.c_str(), false);
    else
        SDL_ShowSaveFileDialog(FileChosen, data, w, kind == 3 ? text : json, 1, initial.c_str());
#endif
}
}  // namespace din::studio
