use primus_glwe::{
    GlevParameters, GlweParameters, GlweSecretKey, NttGadgetEncryptContext,
    NttGlweSchemeSwitchContext, NttGlweSchemeSwitchKey, NttGlweSecretKey, SecretKeyDistr,
};
use primus_lattice::{
    context::NttGlweExternalProductContext,
    ggsw::NttGgsw,
    glev::NttGlev,
    glwe::{Glwe, NttGlwe},
};
use primus_modulus::BarrettModulus;
use primus_ntt::{NttTable, U64NttTable};
use primus_poly::Polynomial;
use rand::{SeedableRng, rngs::StdRng};

const POLY_LENGTH: usize = 256;
const DIMENSION: usize = 2;
const PLAINTEXT_MODULUS: u64 = 16;
const MODULUS: u64 = 1_125_899_906_826_241;

#[test]
fn ntt_scheme_switch_produces_an_external_product_control() {
    let modulus = BarrettModulus::new(MODULUS);
    let ntt = U64NttTable::new(POLY_LENGTH.trailing_zeros(), modulus).unwrap();
    let glwe_parameters = GlweParameters::new(
        DIMENSION,
        POLY_LENGTH,
        PLAINTEXT_MODULUS,
        modulus,
        SecretKeyDistr::UniformBinary,
        0.7,
    );
    let output_parameters = GlevParameters::with_glwe_params(&glwe_parameters, 10, Some(2));
    let scheme_parameters = GlevParameters::with_glwe_params(&glwe_parameters, 10, Some(3));
    let mut rng = StdRng::seed_from_u64(0x0043_4253_5052_494d);
    let coefficient_secret = GlweSecretKey::generate(
        glwe_parameters.size(),
        glwe_parameters.secret_key_sampler(),
        &mut rng,
    );
    let secret = NttGlweSecretKey::from_coeff_secret_key(&coefficient_secret, &ntt);
    let mut gadget = NttGadgetEncryptContext::new(scheme_parameters.size());

    let scheme_key = NttGlweSchemeSwitchKey::generate(
        &coefficient_secret,
        &secret,
        output_parameters.size(),
        &scheme_parameters,
        &ntt,
        &mut rng,
        &mut gadget,
    );
    gadget.resize(output_parameters.size());
    let mut control_message = vec![0; POLY_LENGTH];
    control_message[0] = 1;
    let mut input_glev: NttGlev<Vec<u64>> = NttGlev::zero(output_parameters.glev_len());
    secret.encrypt_glev_to(
        &Polynomial::new(control_message),
        &mut input_glev,
        &output_parameters,
        &ntt,
        &mut rng,
        &mut gadget,
    );
    let input_glev = input_glev.into_coeff_form(&ntt);
    let mut control: NttGgsw<Vec<u64>> = NttGgsw::zero(output_parameters.ggsw_len());
    let mut scheme_context = NttGlweSchemeSwitchContext::new(scheme_parameters.size());
    scheme_key.apply_to(
        &input_glev,
        &mut control,
        modulus,
        &ntt,
        &mut scheme_context,
    );

    let selected_message = vec![5; POLY_LENGTH];
    let mut selected: NttGlwe<Vec<u64>> = NttGlwe::zero(glwe_parameters.glwe_len());
    secret.encrypt_to(
        &Polynomial::new(selected_message.as_slice()),
        &mut selected,
        &glwe_parameters,
        &ntt,
        &mut rng,
    );
    let selected = selected.into_coeff_form(&ntt);
    let mut product: Glwe<Vec<u64>> = Glwe::zero(glwe_parameters.glwe_len());
    let mut external_product = NttGlweExternalProductContext::new(output_parameters.size());
    control.external_product_to(
        &selected,
        &mut product,
        output_parameters.basis(),
        modulus,
        &ntt,
        &mut external_product,
    );
    let product = product.into_ntt_form(&ntt);
    assert_eq!(
        secret.decrypt(&product, &glwe_parameters, &ntt).as_ref(),
        selected_message
    );

    // Reject mismatched modulus/table pairs before copying any row.
    let wrong_modulus = BarrettModulus::new(132_120_577u64);
    let wrong_table = U64NttTable::new(POLY_LENGTH.trailing_zeros(), wrong_modulus).unwrap();
    let wrong_length_table = U64NttTable::new((POLY_LENGTH * 2).trailing_zeros(), modulus).unwrap();
    for (modulus, table) in [
        (modulus, &wrong_table),
        (wrong_modulus, &wrong_table),
        (modulus, &wrong_length_table),
    ] {
        let mut output = NttGgsw::new(vec![7u64; output_parameters.ggsw_len()]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scheme_key.apply_to(
                    &input_glev,
                    &mut output,
                    modulus,
                    table,
                    &mut scheme_context,
                );
            }))
            .is_err()
        );
        assert_eq!(output.as_ref(), vec![7u64; output_parameters.ggsw_len()]);
    }
}
