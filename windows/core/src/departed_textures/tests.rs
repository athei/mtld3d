use super::DepartedTextures;
use crate::ids::TextureId;

#[test]
fn a_fresh_list_drains_nothing() {
    let departed = DepartedTextures::default();
    assert!(departed.take().is_empty());
}

#[test]
fn filed_textures_drain_once_in_filing_order() {
    let departed = DepartedTextures::default();
    let first = TextureId::new_unique();
    let second = TextureId::new_unique();
    departed.note(first);
    departed.note(second);
    assert_eq!(departed.take(), vec![first, second]);
    assert!(
        departed.take().is_empty(),
        "a drained id is not drained again"
    );
}

#[test]
fn a_texture_filed_twice_drains_once() {
    let departed = DepartedTextures::default();
    let id = TextureId::new_unique();
    departed.note(id);
    departed.note(id);
    assert_eq!(departed.take(), vec![id]);
}

#[test]
fn a_texture_that_moved_back_is_not_drained() {
    let departed = DepartedTextures::default();
    let returned = TextureId::new_unique();
    let gone = TextureId::new_unique();
    departed.note(returned);
    departed.note(gone);
    departed.cancel(returned);
    assert_eq!(departed.take(), vec![gone]);
}

#[test]
fn cancelling_the_last_filed_texture_leaves_nothing_to_drain() {
    let departed = DepartedTextures::default();
    let id = TextureId::new_unique();
    departed.note(id);
    departed.cancel(id);
    assert!(departed.take().is_empty());
    departed.cancel(id);
    assert!(
        departed.take().is_empty(),
        "a cancel with nothing filed is a no-op"
    );
}

#[test]
fn a_texture_filed_after_a_drain_reaches_the_next_one() {
    let departed = DepartedTextures::default();
    let first = TextureId::new_unique();
    departed.note(first);
    assert_eq!(departed.take(), vec![first]);
    let later = TextureId::new_unique();
    departed.note(later);
    assert_eq!(departed.take(), vec![later]);
}

#[test]
fn moves_filed_from_other_threads_all_reach_the_drain() {
    let departed = DepartedTextures::default();
    let ids: Vec<TextureId> = (0..64).map(|_| TextureId::new_unique()).collect();
    std::thread::scope(|scope| {
        for chunk in ids.chunks(16) {
            let departed = &departed;
            scope.spawn(move || {
                for &id in chunk {
                    departed.note(id);
                }
            });
        }
    });
    let mut drained = departed.take();
    drained.sort_by_key(|id| id.raw());
    let mut expected = ids;
    expected.sort_by_key(|id| id.raw());
    assert_eq!(drained, expected);
}
