{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc

    # Build system
    cmake

    # Libraries
    boost
    jemalloc
    pkg-config
    openssl
    icu
    zstd
  ];
}
