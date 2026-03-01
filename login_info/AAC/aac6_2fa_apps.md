# AAC6 2FA Apps — Recommended Authenticator Apps for Two-Factor Authentication

## Description

The AAC6 cluster uses TOTP (Time-based One-Time Password) two-factor authentication. Any app that supports the TOTP standard (RFC 6238) will work. This guide covers recommended apps for phones, smartwatches, and computers to help you choose the best option for your workflow.

When you first log in, a QR code is displayed in your terminal. Scan it with any of the apps listed below. The app will then generate a 6-digit code that changes every 30 seconds. Enter this code when prompted during SSH login.

## Choosing an App

| App                      | Phone   | Watch | Computer |
|--------------------------|---------|-------|----------|
| Authy                    | Yes     | Yes   | No       |
| Google Authenticator     | Yes     | Yes   | No       |
| Microsoft Authenticator  | Yes     | Yes   | No       |
| 2FAS                     | Yes     | Yes   | Yes*     |
| Bitwarden                | Yes     | No    | Yes      |
| 1Password                | Yes     | No    | Yes      |
| KeePassXC                | No      | No    | Yes      |
| OTP Auth                 | iOS     | Yes   | Mac      |
| Aegis                    | Android | No    | No       |
| FreeOTP                  | Yes     | No    | No       |

\* 2FAS computer support is via browser extension, not a standalone app.

---

## Phone Apps

All of these apps are free unless noted otherwise.

### Authy (Twilio) — Recommended

**Platforms:** Android, iOS

Authy is a polished TOTP app with encrypted cloud backup of your tokens. If you lose your phone, your tokens can be restored on a new device. It also supports Apple Watch and Wear OS (see [Watch Apps](#watch-apps) below).

- Android: Google Play Store → search "Twilio Authy"
- iOS: App Store → search "Twilio Authy"

Authy previously offered desktop apps for Windows, Mac, and Linux, but those were discontinued in March 2024. Use one of the desktop alternatives listed under [Computer Apps](#computer-apps) if you need TOTP on your workstation.

### Google Authenticator

**Platforms:** Android, iOS

Google Authenticator is the most widely used TOTP app. It supports cloud backup through your Google account and can transfer tokens between devices via QR code export.

- Android: Google Play Store → search "Google Authenticator"
- iOS: App Store → search "Google Authenticator"

Google Authenticator also supports Wear OS smartwatches (see [Watch Apps](#watch-apps)).

### Microsoft Authenticator

**Platforms:** Android, iOS

Microsoft Authenticator supports TOTP codes alongside Microsoft account push notifications. It offers cloud backup via your Microsoft account.

- Android: Google Play Store → search "Microsoft Authenticator"
- iOS: App Store → search "Microsoft Authenticator"

Supports Apple Watch for code viewing (see [Watch Apps](#watch-apps)).

### 2FAS

**Platforms:** Android, iOS

2FAS is a free, open-source authenticator with a clean interface. It supports cloud sync and has companion browser extensions for Chrome and Firefox that let you fill TOTP codes from your phone without typing them.

- Android: Google Play Store → search "2FAS Authenticator"
- iOS: App Store → search "2FAS Authenticator"
- Browser extension: <https://2fas.com>

Supports Apple Watch (see [Watch Apps](#watch-apps)).

### FreeOTP

**Platforms:** Android, iOS

FreeOTP is a lightweight, open-source authenticator originally developed by Red Hat. It stores tokens locally on the device with no cloud sync. Good for users who prefer minimal apps with no account required.

- Android: Google Play Store or F-Droid → search "FreeOTP"
- iOS: App Store → search "FreeOTP"

### Aegis Authenticator (Android only)

**Platforms:** Android

Aegis is a free, open-source authenticator for Android with strong security features including encrypted vault storage and biometric unlock. Supports encrypted backups and import from most other authenticator apps.

- Android: Google Play Store or F-Droid → search "Aegis Authenticator"

### OTP Auth (Apple only)

**Platforms:** iOS, Mac, Apple Watch

OTP Auth is a native Apple ecosystem authenticator that syncs tokens across iPhone, iPad, Mac, and Apple Watch via iCloud. The Mac app provides native desktop TOTP without needing a browser extension.

- iOS/Mac: App Store → search "OTP Auth"

The basic version is free; the pro version (one-time purchase) adds iCloud sync and Apple Watch support.

---

## Watch Apps

Smartwatch apps let you read your 6-digit TOTP code from your wrist without pulling out your phone. This is especially convenient for frequent SSH logins.

### Apple Watch

- **Authy** — Companion to the Authy iOS app. Syncs all tokens to your watch automatically.
- **Microsoft Authenticator** — Shows codes on your Apple Watch when the iOS app is installed.
- **2FAS** — Displays codes on Apple Watch, synced from the iOS app.
- **OTP Auth** — Native Apple Watch app with iCloud sync (pro version).

### Wear OS (Android watches)

- **Authy** — Companion for Wear OS watches. Syncs tokens from the Android app.
- **Google Authenticator** — Wear OS companion shows your codes on compatible watches.

---

## Computer Apps

Desktop and browser-based authenticators are useful when you are already at your workstation and do not want to reach for your phone.

### KeePassXC — Recommended for Desktop

**Platforms:** Windows, Mac, Linux

KeePassXC is a free, open-source password manager that includes a built-in TOTP generator. Tokens are stored in your encrypted KeePass database. You can add TOTP to any entry by pasting the secret key or scanning a QR code (via screen capture).

- Download: <https://keepassxc.org>
- Linux: also available via `apt install keepassxc` or Flatpak/Snap

To add a TOTP token in KeePassXC:

1. Create a new entry or open an existing one.
2. Go to the "TOTP" section and click "Set up TOTP".
3. Paste the secret key shown during enrollment.
4. Codes appear via right-click → "TOTP" → "Show TOTP" or Ctrl+T.

### Bitwarden

**Platforms:** Windows, Mac, Linux, Browser extensions

Bitwarden is a popular open-source password manager. The premium plan ($10/year) includes a built-in TOTP generator that works across all platforms and browser extensions.

- Download: <https://bitwarden.com>

### 1Password

**Platforms:** Windows, Mac, Linux, Browser extensions

1Password is a commercial password manager (subscription required) with built-in TOTP support. Adding a TOTP token is as simple as pasting the secret key or scanning the QR code from your screen.

- Download: <https://1password.com>

### 2FAS Browser Extension

**Platforms:** Chrome, Firefox (pairs with phone app)

The 2FAS browser extension pairs with the 2FAS phone app. When a site or terminal emulator requests a code, you can approve it from your phone and the extension fills it in. This is not a standalone desktop authenticator but is convenient for browser-based workflows.

- Install: <https://2fas.com>

---

## Setup Instructions

1. Install one of the apps listed above on your preferred device **before** your first SSH login, so you are ready to scan the QR code.
2. SSH into the AAC6 cluster. On your first login, a TOTP token is automatically enrolled and a QR code is displayed in the terminal.
3. Open your authenticator app and scan the QR code. If your terminal does not render the QR code properly, enter the secret key shown below the QR code manually into the app.
4. You will be prompted to verify your setup by entering the 6-digit code from your app. This confirms everything is working before you log out. You can press Enter to skip, but this is not recommended — if your app is misconfigured, you will be unable to log in next time.
5. On all future logins, you will be prompted for your 6-digit code after SSH key authentication.

If you need to re-enroll or switch between TOTP and email-based authentication, run the `enroll-2fa` command. See `man enroll-2fa`.

## Tips

- **Back up your tokens.** Use an app that supports cloud backup (Authy, Google Authenticator, Microsoft Authenticator, 2FAS) or export your tokens to a second device. If you lose your only device, you will need an administrator to reset your 2FA.
- **Use multiple devices.** Consider setting up TOTP on both your phone and your computer (e.g., Google Authenticator on your phone and KeePassXC on your laptop). Both will generate valid codes from the same secret. To do this, scan the QR code with both apps during initial setup, or run `enroll-2fa totp` to generate a new QR code.
- **Check your clock.** TOTP codes depend on accurate time. If your codes are being rejected, make sure your device's clock is synchronized (enable automatic time in your device settings).
- **Smartwatch for convenience.** If you SSH in frequently, a watch app means you never have to unlock your phone just to read a 6-digit code.

## Troubleshooting

**"Invalid code" on every attempt**
Your device clock may be out of sync. On your phone, enable automatic date/time in Settings. On desktop, ensure NTP is enabled. Some authenticator apps (like Google Authenticator) have a "Time correction for codes" option in their settings.

**QR code does not display properly**
Make sure your terminal window is wide enough (at least 80 columns) and supports UTF-8. Alternatively, enter the secret key manually into your app.

**Lost your phone or authenticator app**
Contact your system administrator at Bob.Robey@amd.com to have your 2FA token reset. You will then re-enroll on your next login.

**Want to switch to email-based codes**
Run `enroll-2fa email` on the control node. See `man enroll-2fa`.

## See Also

- `man enroll-2fa` — Manage your 2FA tokens
- `man aac6` — AAC6 cluster overview
- `man aac6_novnc` — Browser-based VNC access
