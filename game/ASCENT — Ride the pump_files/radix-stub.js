// Offline stub — no Radix wallet / network auth in local builds.
window._RDT = {
  RadixDappToolkit() {
    return {
      walletApi: {
        provideChallengeGenerator() {},
        async sendOneTimeRequest() {
          return { isErr: () => true, error: new Error("Wallet disabled in offline mode") };
        },
        async sendTransaction() {
          return { isErr: () => true, error: new Error("Shop disabled in offline mode") };
        },
      },
    };
  },
  RadixNetwork: {},
  OneTimeDataRequestBuilder: {
    accounts() {
      return {
        exactly() {
          return { withProof() { return {}; } };
        },
      };
    },
  },
};
